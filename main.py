from data_parser import GameParser
from baseline_model import BaselineModel
import pandas as pd


def main():
    game_parser = GameParser()
    df = game_parser.parse_all_games()

    base_model = BaselineModel(df)
    base_model.evaluate('base_stat_total')
    return df


if __name__ == '__main__':
    main()