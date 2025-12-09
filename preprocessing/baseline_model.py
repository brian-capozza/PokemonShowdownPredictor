class BaselineModel:
    def __init__(self, data):
        self.data = data

    def evaluate(self, method):
        """
        mehods: base_stat_total, p1_win, highest_health
        """
        if method == 'base_stat_total':
            first_turn = self.data.sort_values("turn").groupby("game_id").head(1).copy()

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

            first_turn["p1_total_stats"] = first_turn.apply(lambda r: sum_team(r, "p1"), axis=1)
            first_turn["p2_total_stats"] = first_turn.apply(lambda r: sum_team(r, "p2"), axis=1)

            first_turn["predicted_p1_win"] = (first_turn["p1_total_stats"] > first_turn["p2_total_stats"]).astype(int)

            correct = (first_turn["predicted_p1_win"] == first_turn["p1_win"]).sum()
            total = len(first_turn)
            accuracy = correct / total

            print(f"Base stat method accuracy: {accuracy:.3f} ({correct}/{total})")

            first_turn[
                ["game_id", "p1_total_stats", "p2_total_stats", "predicted_p1_win", "p1_win"]
            ]

            return accuracy
        
        elif method == 'p1_win':
            first_turn = self.data.sort_values("turn").groupby("game_id").head(1).copy()

            first_turn["predicted_p1_win"] = 1

            correct = (first_turn["predicted_p1_win"] == first_turn["p1_win"]).sum()
            total = len(first_turn)
            accuracy = correct / total

            print(f"p1 win method accuracy: {accuracy:.3f} ({correct}/{total})")

            first_turn[
                ["game_id", "predicted_p1_win", "p1_win"]
            ]

            return accuracy
        
        elif method == 'highest_health':
            last_turn = self.data.sort_values("turn").groupby("game_id").tail(1).copy()

            def sum_team(row, prefix):
                total = 0
                for slot in range(1, 7):
                    total += row[f"{prefix}_slot{slot}_current_health"]
                return total

            last_turn["p1_total_health"] = last_turn.apply(lambda r: sum_team(r, "p1"), axis=1)
            last_turn["p2_total_health"] = last_turn.apply(lambda r: sum_team(r, "p2"), axis=1)

            last_turn["predicted_p1_win"] = (last_turn["p1_total_health"] > last_turn["p2_total_health"]).astype(int)

            correct = (last_turn["predicted_p1_win"] == last_turn["p1_win"]).sum()
            total = len(last_turn)
            accuracy = correct / total

            print(f"Highest health method accuracy: {accuracy:.3f} ({correct}/{total})")

            last_turn[
                ["game_id", "p1_total_health", "p2_total_health", "predicted_p1_win", "p1_win"]
            ]

            return accuracy