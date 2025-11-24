from data_parser import GameParser
from auto_encoder import PokemonAutoEncoder, MoveAutoEncoder, embed_column
import pandas as pd


def main():
    game_parser = GameParser()
    df = game_parser.parse_all_games()
    move_ae = MoveAutoEncoder(game_parser.get_move_stats())
    move_emb = move_ae.get_embeddings()
    move_ae.plot_embeddings()

    df = embed_column(df, "move_p1", move_emb, prefix="move_p1")
    df = embed_column(df, "move_p2", move_emb, prefix="move_p2")

    pokemon_ae = PokemonAutoEncoder(game_parser.get_poke_stats())
    pokemon_emb = pokemon_ae.get_embeddings()
    pokemon_ae.plot_embeddings()

    df = embed_column(df, "active_p1", pokemon_emb, prefix="active_p1")
    df = embed_column(df, "active_p2", pokemon_emb, prefix="active_p2")

    weather_ohe = pd.get_dummies(df["weather"], prefix="weather")
    df = pd.concat([df.drop(columns=["weather"]), weather_ohe], axis=1)

    print(df)

    return df


if __name__ == '__main__':
    main()