class BaselineModel:
    def __init__(self, data):
        self.data = data

    def evaluate(self, method):
        """
        mehods: base_stat_total, p1_win, highest_health
        """
        if method == 'base_stat_total':
            # --- Step 1: Keep only 1 row per game (first turn) ---
            first_turn = self.data.sort_values("turn").groupby("game_id").head(1).copy()

            # Helper to sum stats for one team
            def sum_team(row, prefix):
                total = 0
                for slot in range(1, 7):
                    total += (
                        row[f"{prefix}_slot{slot}_hp"] +
                        row[f"{prefix}_slot{slot}_atk"] +
                        row[f"{prefix}_slot{slot}_def"] +
                        row[f"{prefix}_slot{slot}_spa"] +
                        row[f"{prefix}_slot{slot}_spd"] +
                        row[f"{prefix}_slot{slot}_spe"]
                    )
                return total

            # --- Step 2: Compute total base stats ---
            first_turn["p1_total_stats"] = first_turn.apply(lambda r: sum_team(r, "p1"), axis=1)
            first_turn["p2_total_stats"] = first_turn.apply(lambda r: sum_team(r, "p2"), axis=1)

            # --- Step 3: Predict winner ---
            # P1 wins if their team has higher total stats
            first_turn["predicted_p1_win"] = (first_turn["p1_total_stats"] > first_turn["p2_total_stats"]).astype(int)

            # --- Step 4: Compare prediction to actual winner ---
            correct = (first_turn["predicted_p1_win"] == first_turn["p1_win"]).sum()
            total = len(first_turn)
            accuracy = correct / total

            print(f"Base stat method accuracy: {accuracy:.3f} ({correct}/{total})")

            # Return detailed results
            first_turn[
                ["game_id", "p1_total_stats", "p2_total_stats", "predicted_p1_win", "p1_win"]
            ]

            return accuracy
        
        elif method == 'p1_win':
            # --- Step 1: Keep only 1 row per game (first turn) ---
            first_turn = self.data.sort_values("turn").groupby("game_id").head(1).copy()

            # --- Step 3: Predict winner ---
            first_turn["predicted_p1_win"] = 1

            # --- Step 4: Compare prediction to actual winner ---
            correct = (first_turn["predicted_p1_win"] == first_turn["p1_win"]).sum()
            total = len(first_turn)
            accuracy = correct / total

            print(f"p1 win method accuracy: {accuracy:.3f} ({correct}/{total})")

            # Return detailed results
            first_turn[
                ["game_id", "predicted_p1_win", "p1_win"]
            ]

            return accuracy
        
        elif method == 'highest_health':
            # --- Step 1: Keep only 1 row per game (last turn) ---
            last_turn = self.data.sort_values("turn").groupby("game_id").tail(1).copy()

            # Helper to sum stats for one team
            def sum_team(row, prefix):
                total = 0
                for slot in range(1, 7):
                    total += row[f"{prefix}_slot{slot}_current_health"]
                return total

            # --- Step 2: Compute total base stats ---
            last_turn["p1_total_health"] = last_turn.apply(lambda r: sum_team(r, "p1"), axis=1)
            last_turn["p2_total_health"] = last_turn.apply(lambda r: sum_team(r, "p2"), axis=1)

            # --- Step 3: Predict winner ---
            # P1 wins if their team has higher total stats
            last_turn["predicted_p1_win"] = (last_turn["p1_total_health"] > last_turn["p2_total_health"]).astype(int)

            # --- Step 4: Compare prediction to actual winner ---
            correct = (last_turn["predicted_p1_win"] == last_turn["p1_win"]).sum()
            total = len(last_turn)
            accuracy = correct / total

            print(f"Highest health method accuracy: {accuracy:.3f} ({correct}/{total})")

            # Return detailed results
            last_turn[
                ["game_id", "p1_total_health", "p2_total_health", "predicted_p1_win", "p1_win"]
            ]

            return accuracy