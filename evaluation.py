import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import precision_score, recall_score, f1_score

def set_pokemon_theme():
    mpl.rcParams.update({
        "figure.facecolor": "#FAF9F6",
        "axes.facecolor": "#FAF9F6",
        "axes.edgecolor": "#1A1A1A",
        "axes.labelcolor": "#1A1A1A",
        "xtick.color": "#1A1A1A",
        "ytick.color": "#1A1A1A",
        "grid.color": "#D0D0D0",
        "font.size": 12,
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "lines.linewidth": 2.5,
        "savefig.facecolor": "#FAF9F6",
        "savefig.edgecolor": "#FAF9F6",
    })


class ModelEvaluation:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.seq_builder = None

    def _unpack_batch(self, batch): # Used ChatGPT to help normalize the class for both NN models
        if len(batch) == 3:
            x, lengths, y = batch
        elif len(batch) == 2:
            x, y = batch
            lengths = torch.ones(len(y), dtype=torch.long, device=x.device)
        else:
            raise ValueError(f"Unexpected batch format with len={len(batch)}")

        return x, lengths, y

    def _forward_model(self, x, lengths): # Used ChatGPT to help normalize the class for both NN models
        if hasattr(self.model, "network"):
            if x.dim() == 2:
                return self.model(x)
            elif x.dim() == 3:
                B, T, F = x.shape
                last = x[torch.arange(B, device=x.device), lengths - 1, :]
                return self.model(last)
            else:
                raise ValueError(f"Unexpected input dim for MLP: x.dim() = {x.dim()}")
        try:
            return self.model(x, lengths)
        except TypeError:
            return self.model(x)

    def _evaluate(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in loader:
                x, lengths, y = self._unpack_batch(batch)
                x, y = x.to(self.device), y.to(self.device)
                logits = self._forward_model(x, lengths)
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
            for batch in loader:
                x, lengths, y = self._unpack_batch(batch)
                x, y = x.to(self.device), y.to(self.device)
                logits = self._forward_model(x, lengths)
                sample_losses = self.criterion(logits, y)
                total_loss += sample_losses.mean().item()
                count += 1
        return total_loss / max(1, count)

    def _evaluate_with_prefix(self, loader, prefix_frac=0.5):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for seqs, labels in loader.dataset:
                seq_len = len(seqs)
                prefix_len = max(1, int(prefix_frac * seq_len))

                seq_prefix = seqs[:prefix_len].unsqueeze(0).to(self.device)
                lengths = torch.tensor([prefix_len], dtype=torch.long).to(self.device)

                logits = self._forward_model(seq_prefix, lengths)
                pred = (torch.sigmoid(logits) > 0.5).int().item()
                correct += (pred == labels.item())
                total += 1

        return correct / total if total > 0 else 0.0

    def train_and_evaluate(
        self,
        train_loader,
        val_loader,
        test_loader,
        seq_builder,
        epochs,
        prefix_min_start,
        prefix_min_end,
        label_smoothing,
        patience,
        grad_noise_std,
    ):
        self.test_loader = test_loader
        self.seq_builder = seq_builder

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

            # update prefix curriculum only if we have a seq_builder
            if self.seq_builder is not None: # Used ChatGPT to help normalize the class for both NN models
                progress = (epoch - 1) / max(1, epochs - 1)
                current_prefix_min_frac = (
                    prefix_min_start
                    + progress * (prefix_min_end - prefix_min_start)
                )
                self.seq_builder.set_prefix_frac(current_prefix_min_frac)
                current_prefix_min_frac = float(np.clip(current_prefix_min_frac, 0.0, 1.0))
            else:
                current_prefix_min_frac = 1.0

            for batch in train_loader:
                x, lengths, y = self._unpack_batch(batch)
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                logits = self._forward_model(x, lengths)

                if label_smoothing > 0.0: # Used ChatGPT to help improve models
                    y_smooth = y * (1.0 - label_smoothing) + 0.5 * label_smoothing
                else:
                    y_smooth = y

                sample_losses = self.criterion(logits, y_smooth)

                if lengths is not None and lengths.numel() > 0:
                    w = (lengths.float() / lengths.max().float()).to(self.device)
                else:
                    w = torch.ones_like(sample_losses)

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
        # Used ChatGPT to help understand how to calculate feature importance for gradients
        was_training = self.model.training
        self.model.train()

        def disable_dropout(module):
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.model.apply(disable_dropout)

        importance = None
        batches_done = 0

        for batch in test_loader:
            x, lengths, y = self._unpack_batch(batch)
            x = x.to(self.device).requires_grad_(True)
            lengths = lengths.to(self.device)

            logits = self._forward_model(x, lengths)
            probs = torch.sigmoid(logits)

            loss = probs.mean()
            self.model.zero_grad()
            loss.backward()

            if x.dim() == 3:
                grad = x.grad.detach().abs().mean(dim=(0, 1))
            else:
                grad = x.grad.detach().abs().mean(dim=0)

            if importance is None:
                importance = grad.cpu().numpy()
            else:
                importance += grad.cpu().numpy()

            batches_done += 1
            if batches_done >= num_batches:
                break

        importance /= max(1, batches_done)

        if was_training:
            self.model.train()
        else:
            self.model.eval()

        return importance

    def plot_feature_importance(self, importance, feature_names, top_k=30): # Used ChatGPT to help build visualizations
        idx = np.argsort(importance)[::-1][:top_k]
        top_features = np.array(feature_names)[idx]
        top_values = importance[idx]

        plt.figure(figsize=(10, 12))
        plt.barh(top_features, top_values)
        plt.gca().invert_yaxis()
        plt.title(f"Top {top_k} Most Important Features (Gradient-Based)")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=250, bbox_inches="tight")
        plt.show()

    def _compute_metrics(self, loader):
        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                x, lengths, y = self._unpack_batch(batch)
                x, y = x.to(self.device), y.to(self.device)
                logits = self._forward_model(x, lengths)
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

    def visualize_training(self): # Used ChatGPT to help build visualizations
        set_pokemon_theme()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Pokemon blue & yellow
        blue = "#3B4CCA"
        yellow = "#FFCB05"

        ax1.plot(self.history["epoch"], self.history["train_loss"], label="Train Loss",
                color=blue, marker="o")
        ax1.plot(self.history["epoch"], self.history["val_loss"], label="Val Loss",
                color=yellow, marker="s")
        ax1.set_title("Training vs Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(alpha=0.35)
        ax1.legend()

        ax2.plot(self.history["epoch"], self.history["train_acc"], label="Train Acc",
                color=blue, marker="o")
        ax2.plot(self.history["epoch"], self.history["val_acc"], label="Val Acc",
                color=yellow, marker="s")
        ax2.axhline(self.best_val_acc_at_best_loss, linestyle="--", color="#1A1A1A", alpha=0.4,
                    label=f"Best Val Acc ({self.best_val_acc_at_best_loss:.3f})")
        ax2.set_title("Training vs Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.grid(alpha=0.35)
        ax2.legend()

        plt.tight_layout()
        plt.savefig("training_history.png", dpi=250, bbox_inches="tight")
        plt.show()


    def visualize_games(self, game_test, test_seqs, test_labels): # Used ChatGPT to help build visualizations
        set_pokemon_theme()
        print("\nGenerating Pokémon-style win probability curves...\n")

        self.model.eval()
        ids = np.unique(game_test) if isinstance(game_test, np.ndarray) else game_test.unique()

        n_games = min(12, len(test_seqs))
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        axes = axes.flatten()

        green = "#4CAF50"
        red = "#CC0000"
        hud_gray = "#3A3A3A"

        for idx in range(n_games):
            seq = test_seqs[idx].to(self.device)
            true_label = test_labels[idx].item()
            game_id = ids[idx]
            T = seq.shape[0]

            probs = []
            for t in range(1, T + 1):
                with torch.no_grad():
                    if hasattr(self.model, "network"):
                        logits = self._forward_model(seq[t - 1].unsqueeze(0), None)
                    else:
                        logits = self._forward_model(seq[:t].unsqueeze(0), torch.tensor([t]))
                probs.append(torch.sigmoid(logits).item())

            ax = axes[idx]
            final = probs[-1]
            correct = (final > 0.5) == (true_label == 1)
            color = green if correct else red

            # Pokémon HP-bar look
            ax.plot(probs, color=color)
            ax.fill_between(range(len(probs)), 0, probs, color=color, alpha=0.25)

            # Gray dashed midline (Pokémon-style HUD)
            ax.axhline(0.5, color=hud_gray, linestyle="--", alpha=0.5)

            winner = "P1" if true_label == 1 else "P2"
            status = "✓" if correct else "✗"
            ax.set_title(f"Game {game_id} — Winner {winner} {status}\nFinal: {final:.2f}",
                        fontsize=10, fontweight="bold")

            ax.set_xlabel("Turn")
            ax.set_ylabel("P1 Win Probability")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.suptitle("Pokémon Win Probability Curves", fontsize=18, weight="bold")
        plt.tight_layout(rect=(0, 0, 1, 0.95))
        plt.savefig("win_curves_grid.png", dpi=250, bbox_inches="tight")
        plt.show()


    def visualize_prf(self): # Used ChatGPT to help build visualizations
        set_pokemon_theme()

        fig, ax = plt.subplots(figsize=(7, 5))

        metrics = ["Precision", "Recall", "F1 Score"]
        values = [float(self.test_precision), float(self.test_recall), float(self.test_f1)]

        colors = ["#3B4CCA", "#FFCB05", "#CC0000"]  # blue, yellow, red

        bars = ax.bar(metrics, values, color=colors, edgecolor="#1A1A1A", linewidth=1.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val:.3f}", ha="center", fontsize=12,
                    weight="bold")

        ax.set_ylim(0, 1.15)
        ax.set_title("Pokémon Model Performance Metrics", fontsize=16)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig("prf_metrics.png", dpi=250, bbox_inches="tight")
        plt.show()


    def summary(self): # Used ChatGPT to help display summary statistics
        print("\nGenerating calibration and prediction distribution plots...")

        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in self.test_loader:
                x, lengths, y = self._unpack_batch(batch)
                x, y = x.to(self.device), y.to(self.device)
                logits = self._forward_model(x, lengths)
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
