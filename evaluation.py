import torch

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score

class ModelEvaluation:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device


    def _evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, lengths, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x, lengths)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()
                correct += (preds == y.int()).sum().item()
                total += len(y)
        return correct / total if total > 0 else 0.0


    def _evaluate_loss(self, loader):
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for x, lengths, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x, lengths)
                # Use true labels for validation loss (no smoothing)
                sample_losses = self.criterion(logits, y)
                total_loss += sample_losses.mean().item()
                count += 1
        return total_loss / max(1, count)


    def train_and_evaluate(self, train_loader, val_loader, test_loader, seq_builder, epochs, prefix_min_start, prefix_min_end, label_smoothing, patience, grad_noise_std):
        self.test_loader = test_loader
        
        best_val_loss = float("inf")
        self.best_val_acc_at_best_loss = 0.0
        patience_counter = 0
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_loss": [],
        }

        print("\nStarting training...\n")

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            batch_count = 0

            # update prefix curriculum (module-level variable)
            progress = (epoch - 1) / max(1, epochs - 1)
            current_prefix_min_frac = (
                prefix_min_start
                + progress * (prefix_min_end - prefix_min_start)
            )
            seq_builder.set_prefix_frac(current_prefix_min_frac)
            current_prefix_min_frac = float(np.clip(current_prefix_min_frac, 0.0, 1.0))

            for x, lengths, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(x, lengths)

                # Label smoothing for training loss
                if label_smoothing > 0.0:
                    y_smooth = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
                else:
                    y_smooth = y

                sample_losses = self.criterion(logits, y_smooth)

                # Prefix-length-based weights (longer prefixes closer to full game)
                w = (lengths.float() / lengths.max().float()).to(self.device)
                loss = (w * sample_losses).mean()

                loss.backward()

                if grad_noise_std > 0:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.add_(grad_noise_std * torch.randn_like(p.grad))

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            train_loss_epoch = total_loss / max(1, batch_count)
            train_acc = self._evaluate(train_loader)
            val_acc = self._evaluate(val_loader)
            val_loss = self._evaluate_loss(val_loader)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss_epoch)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_loss"].append(val_loss)

            print(
                f"Epoch {epoch:2d} | Loss {train_loss_epoch:7.4f} | "
                f"Train {train_acc:.4f} | Val {val_acc:.4f} | ValLoss {val_loss:.4f} "
                f"| prefix_min_frac={current_prefix_min_frac:.2f}"
            )

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                self.best_val_acc_at_best_loss = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⚠ Early stop at epoch {epoch} (no val loss improvement)")
                    break


        state_dict = torch.load("best_model.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)

        val_acc_final = self._evaluate(val_loader)
        self.test_acc = self._evaluate(test_loader)
        self.test_precision, self.test_recall, self.test_f1 = self._compute_metrics(test_loader)

        print(
            f"\nTest Accuracy: {self.test_acc:.4f} | "
            f"Best Val Acc (at best loss): {self.best_val_acc_at_best_loss:.4f} | "
            f"Best Val Loss: {best_val_loss:.4f}"
        )

    def compute_feature_importance(self, test_loader, feature_names, num_batches=20):
        """
        Computes global feature importance by averaging gradients of the output
        with respect to the inputs across several batches.
        """

        # Store original mode
        was_training = self.model.training  

        # We need training mode for cuDNN RNN backward()
        self.model.train()

        # Disable dropout and stochastic depth manually
        def disable_dropout(module):
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.model.apply(disable_dropout)

        importance = None
        batches_done = 0

        for x, lengths, y in test_loader:
            x = x.to(self.device).requires_grad_(True)
            lengths = lengths.to(self.device)

            logits = self.model(x, lengths)
            probs = torch.sigmoid(logits)

            loss = probs.mean()
            self.model.zero_grad()
            loss.backward()

            grad = x.grad.detach().abs().mean(dim=(0, 1))  # [F]

            if importance is None:
                importance = grad.cpu().numpy()
            else:
                importance += grad.cpu().numpy()

            batches_done += 1
            if batches_done >= num_batches:
                break

        importance /= batches_done

        # Restore original mode
        if was_training:
            self.model.train()
        else:
            self.model.eval()

        return importance

    
    def plot_feature_importance(self, importance, feature_names, top_k=30):
        # Sort features by importance
        idx = np.argsort(importance)[::-1][:top_k]
        top_features = np.array(feature_names)[idx]
        top_values = importance[idx]

        plt.figure(figsize=(10, 12))
        plt.barh(top_features, top_values)
        plt.gca().invert_yaxis()
        plt.title(f"Top {top_k} Most Important Features (Gradient-Based)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()



    def _compute_metrics(self, loader):
        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for x, lengths, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x, lengths)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        preds = (all_probs > 0.5).astype(int)

        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)

        return precision, recall, f1


    def visualize_training(self):
        # =========================
        # TRAINING HISTORY PLOTS
        # =========================

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(self.history["epoch"], self.history["train_loss"], label="Train Loss",
                linewidth=2.0, marker="o")
        ax1.plot(self.history["epoch"], self.history["val_loss"], label="Val Loss",
                linewidth=2.0, marker="s")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training vs Validation Loss")
        ax1.grid(alpha=0.3)
        ax1.legend()

        ax2.plot(self.history["epoch"], self.history["train_acc"], label="Train Acc",
                linewidth=2.0, marker="o")
        ax2.plot(self.history["epoch"], self.history["val_acc"], label="Val Acc",
                linewidth=2.0, marker="s")
        ax2.axhline(self.best_val_acc_at_best_loss, linestyle="--", color="gray", alpha=0.6,
                    label=f"Best Val Acc @ Best Loss ({self.best_val_acc_at_best_loss:.3f})")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Train vs Val Accuracy")
        ax2.set_ylim(0.4, 1.0)
        ax2.grid(alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.savefig("training_history.png", dpi=200, bbox_inches="tight")
        plt.show()


    def visualize_games(self, game_test, test_seqs, test_labels):
        print("\nGenerating win probability curves for sample test games...")

        self.model.eval()
        test_game_ids_unique = game_test.unique()

        n_games = min(12, len(test_seqs))

        # Smaller figure so subplots have consistent spacing
        fig, axes = plt.subplots(3, 4, figsize=(18, 11))
        axes = axes.flatten()

        for idx in range(n_games):
            seq = test_seqs[idx].unsqueeze(0).to(self.device)
            true_label = test_labels[idx].item()
            game_id = test_game_ids_unique[idx]

            T = seq.shape[1]
            probs = []
            for t in range(1, T + 1):
                with torch.no_grad():
                    length_t = torch.tensor([t], dtype=torch.long).to(self.device)
                    prob = torch.sigmoid(self.model(seq[:, :t, :], length_t)).item()
                probs.append(prob)

            ax = axes[idx]
            final_prob = probs[-1]
            final_pred = final_prob > 0.5
            correct = ((final_pred and true_label == 1) or ((not final_pred) and true_label == 0))
            color = "#27AE60" if correct else "#E74C3C"

            # Plot main curve
            ax.plot(probs, linewidth=1.5, color=color)
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
            ax.fill_between(range(len(probs)), 0, probs, alpha=0.18, color=color)

            winner = "P1" if true_label == 1 else "P2"
            status = "✓" if correct else "✗"

            # ---- Smaller, compressed 2-line title ----
            title_text = (
                f"Game {game_id} — Winner: {winner} {status}\n"
                f"Pred: {final_prob:.2f}"
            )

            ax.set_title(title_text, fontsize=8, fontweight="bold")
            ax.title.set_position((0.5, 1.15))  # Lift title upward

            ax.set_xlabel("Turn", fontsize=8)
            ax.set_ylabel("P1 Win Prob", fontsize=8)

            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.25, linestyle=":")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Reduce tick label size
            ax.tick_params(labelsize=8)

        # ---- Global title ----
        plt.suptitle(
            "Win Probability Predictions (Green = Correct, Red = Incorrect)",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )

        # ---- Spacing tuned to prevent any clipping ----
        plt.subplots_adjust(
            top=0.90,
            bottom=0.08,
            left=0.05,
            right=0.98,
            hspace=0.55,
            wspace=0.25
        )

        plt.savefig("win_curves_grid.png", dpi=200, bbox_inches="tight")
        plt.show()


    def visualize_prf(self):
        fig, ax = plt.subplots(figsize=(6, 5))

        metrics = ["Precision", "Recall", "F1 Score"]
        values = [
            float(self.test_precision),
            float(self.test_recall),
            float(self.test_f1)
        ]

        bars = ax.bar(metrics, values, color=["#3498DB", "#E67E22", "#9B59B6"], alpha=0.85)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.02,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold")

        ax.set_ylim(0, 1.15)
        ax.set_title("Precision, Recall, F1 Score (Test Set)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        plt.tight_layout()
        plt.savefig("prf_metrics.png", dpi=200, bbox_inches="tight")
        plt.show()



    def summary(self):
        print("\nGenerating calibration and prediction distribution plots...")

        all_probs = []
        all_labels = []
        with torch.no_grad():
            for x, lengths, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x, lengths)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_probs = []
        bin_accs = []

        for i in range(n_bins):
            m = (all_probs >= bins[i]) & (all_probs < bins[i + 1])
            if m.sum() > 0:
                bin_probs.append(bin_centers[i])
                bin_accs.append(all_labels[m].mean())
        # =========================
        # SUMMARY STATS
        # =========================

        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 60)

        high_conf_mask = (all_probs > 0.7) | (all_probs < 0.3)
        if high_conf_mask.sum() > 0:
            high_conf_acc = (
                (all_probs[high_conf_mask] > 0.5) ==
                all_labels[high_conf_mask]
            ).mean()
        else:
            high_conf_acc = float("nan")

        medium_conf_mask = (all_probs >= 0.4) & (all_probs <= 0.6)
        if medium_conf_mask.sum() > 0:
            medium_conf_acc = (
                (all_probs[medium_conf_mask] > 0.5) ==
                all_labels[medium_conf_mask]
            ).mean()
        else:
            medium_conf_acc = float("nan")

        print("\n" + "=" * 60)
        print(f"Overall Test Accuracy:          {self.test_acc:.2%}")
        print("Classification Metrics (Test Set)")
        print("=" * 60)
        print(f"Precision: {self.test_precision:.4f}")
        print(f"Recall:    {self.test_recall:.4f}")
        print(f"F1 Score:  {self.test_f1:.4f}")
        print("=" * 60)
