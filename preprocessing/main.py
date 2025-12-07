from data_parser import GameParser
from baseline_model import BaselineModel
import pandas as pd
import matplotlib.pyplot as plt

GENERATION = 5
FORMAT = 'ou'
BASELINES = ['base_stat_total', 'p1_win', 'highest_health']


def main():
    game_parser = GameParser(GENERATION, FORMAT)
    df = game_parser.parse_all_games()
    df.to_csv(f'datasets/gen{GENERATION}{FORMAT}_games.csv', index=False)

    print(f'===== Evaluation of (gen{GENERATION}{FORMAT}) =====')
    base_model = BaselineModel(df)

    for baseline in BASELINES:
        acc = base_model.evaluate(baseline)
        print(f"{baseline} accuracy: {acc}")


if __name__ == '__main__':
    main()
