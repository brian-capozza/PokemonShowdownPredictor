import requests
import pandas as pd
import json
import os

class MovesetParser():
    def __init__(self, rerun: bool, generation=5, format='ou'):
        self.game_type = f"gen{generation}{format}"
        self.meta = {}
        self.log = {}
        self.move_corpus = {}
        self.data_loaded = False
        if rerun:
            self._parse_all_games()
        else:
            self._load_movesets()

        
    def _parse_turns(self, game_id, game_text):
        self.log[game_id] = {}
        game_text_split = game_text.split('\n')
        turn_num = None
        for line in game_text_split:
            if '|turn|' in line:
                turn_num = int(line.split('|')[2])
                if turn_num not in self.log[game_id].keys():
                    self.log[game_id][turn_num] = []

            if turn_num is not None and '|turn|' not in line:
                self.log[game_id][turn_num].append(line)


    def _parse_game_metadata(self, game_id, game_text):
        self.meta[game_id] = {}
        game_info = self.meta[game_id]
        game_info['nickname_converter'] = {}
        game_text_split = game_text.split('\n')
        for line in game_text_split:
            if ('|-message|' in line and ('forfeited' in line or 'lost due to inactivity' in line)):
                # |-message|spindakaasie1 forfeited.
                return False

            if '|switch|' in line or '|drag|' in line or '|replace|' in line:
                # |switch|p1a: Pretty|Gardevoir, M|100/100
                parts = line.split("|")

                nickname = parts[2].split(': ')[1]
                pokemon = parts[3]
                if ',' in pokemon:
                    pokemon_name = pokemon.split(',')[0]
                else:
                    pokemon_name = pokemon

                game_info['nickname_converter'][nickname] = pokemon_name
        
        return True
    
    def _parse_game_data(self, game_id, game_text):
        self._parse_game_metadata(game_id, game_text)
        self._parse_turns(game_id, game_text)

        move_dict = {}

        # Iterate through turns
        for turn_num, turn in self.log[game_id].items():
            turn_num = int(turn_num)
            for event in turn:

                # ─────────── MOVE ───────────
                if '|move|' in event:
                    # |move|p1a: Persian|Fake Out|p2a: Landorus
                    event_list = event.split('|')

                    nickname = event_list[2].split(': ')[1]


                    pokemon_name = self.meta[game_id]['nickname_converter'][nickname]

                    move = event_list[3]
                    
                    if pokemon_name not in move_dict.keys():
                        move_dict[pokemon_name] = {}
                    
                    if move not in move_dict[pokemon_name].keys():
                        move_dict[pokemon_name][move] = 1

        return move_dict

    def _parse_all_games(self):
        path = f"datasets/{self.game_type}_game_text.pkl"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Game file not found: {path}")

        df = pd.read_pickle(path)
        for row in df.itertuples():
            move_dict = self._parse_game_data(row.game_id, row.raw_text)
            for pokemon, moves in move_dict.items():
                if pokemon not in self.move_corpus.keys():
                    self.move_corpus[pokemon] = {}
                
                for move in moves.keys():
                    if move not in self.move_corpus[pokemon]:
                        self.move_corpus[pokemon][move] = 1
                    else:
                        self.move_corpus[pokemon][move] += 1


        rows = []

        for pokemon, moves in self.move_corpus.items():
            total_uses = sum(moves.values())

            # Sort moves by usage and take top 4
            top_moves = sorted(moves.items(), key=lambda x: x[1], reverse=True)[:4]

            row = {"pokemon": pokemon}

            for i, (move, count) in enumerate(top_moves, start=1):
                pct = round((count / total_uses) * 100, 2)
                row[f"move_{i}"] = move
                row[f"pct_{i}"] = pct

            rows.append(row)

        self.movesets = pd.DataFrame(rows)
        self.movesets.to_csv(f'datasets/{self.game_type}_move_data.csv')
        self.data_loaded = True

    def _load_movesets(self):
        path = f'datasets/{self.game_type}_move_data.csv'

        if not os.path.exists(path):
            raise FileNotFoundError(f"Moveset file not found: {path}")

        self.movesets = pd.read_csv(path)
        self.data_loaded = True

    def get_movesets(self) -> pd.DataFrame:
        if self.data_loaded:
            return self.movesets
        else:
            raise FileNotFoundError(f"Moveset file not loaded")

class DataCollector():
    def __init__(self, use_api: bool, generation=5, format='ou', num_pages=2):
        '''
        self.games: pd.DataFrame
        self.moves: dict
        self.pokemon: dict
        '''
        self.game_type = f"gen{generation}{format}"
        self.num_pages = num_pages
        if use_api:
            self._build_games()
            self._build_moves()
            self._build_pokemon()
            self.mp = MovesetParser(rerun=True, generation=generation, format=format)
        else:
            self._load_games()
            self._load_moves()
            self._load_pokemon()
            self.mp = MovesetParser(rerun=False, generation=generation, format=format)


    def _build_games(self):
        self.games = {}
        base_url = 'https://replay.pokemonshowdown.com/'
        for page in range(1, self.num_pages + 1):
            games_url = f'https://replay.pokemonshowdown.com/search.json?format={self.game_type}&page={page}&sort=rating'
            data = requests.get(games_url).json()
            for game in data:
                game_data = requests.get(f'{base_url}{game["id"]}.log')
                self.games[game["id"]] = game_data.text

        self._save_games()

        self.games = pd.DataFrame(
            self.games.items(),
            columns=["game_id", "raw_text"]
        )
        
        

    def _build_moves(self):
        self.moves = {}
        moves_url = "https://play.pokemonshowdown.com/data/moves.json"

        response = requests.get(moves_url)
        moves = response.json()

        for move, stats in moves.items():
            if stats.get("accuracy"):
                accuracy = 100
            else:
                accuracy = stats['accuracy']
            self.moves[move.lower()] = {
                'accuracy': accuracy,
                'base_power': stats['basePower'],
                'category': stats['category'],
                'type': stats['type'],
                'pp': stats['pp']
            }
        
        self._save_moves()

    
    def _build_pokemon(self):
        self.pokemon = {}
        pokemon_url = "https://play.pokemonshowdown.com/data/pokedex.json"
        pokedex = requests.get(pokemon_url).json()
        for pokemon, stats in pokedex.items():
            if stats.get('num'):
                stats_types = stats['baseStats']
                stats_types['type1'] = stats['types'][0]
                if len(stats['types']) == 2:
                    stats_types['type2'] = stats['types'][1]
                else:
                    stats_types['type2'] = None

                self.pokemon[pokemon.lower()] = stats_types

        self._save_pokemon()
    
    def _save_pokemon(self):
        with open(f"datasets/{self.game_type}_pokemon.json", "w", encoding="utf-8") as f:
            json.dump(self.pokemon, f, indent=2)
    
    def _save_moves(self):
        with open(f"datasets/{self.game_type}_moves.json", "w", encoding="utf-8") as f:
            json.dump(self.moves, f, indent=2)
    
    def _save_games(self):    
        df = pd.DataFrame(
            self.games.items(),
            columns=["game_id", "raw_text"]
        )
        df.to_pickle(f"datasets/{self.game_type}_game_text.pkl")


    def _load_games(self):
        path = f"datasets/{self.game_type}_game_text.pkl"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Game file not found: {path}")

        self.games = pd.read_pickle(path)


    def _load_pokemon(self):
        path = f"datasets/{self.game_type}_pokemon.json"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Pokemon file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            self.pokemon = json.load(f)


    def _load_moves(self):
        path = f"datasets/{self.game_type}_moves.json"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Moves file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            self.moves = json.load(f)


    def get_pokemon(self, pokemon=None):
        if pokemon is None:
            return self.pokemon
        if 'Gastrodon' in pokemon:
            pokemon = 'Gastrodon'
        pokemon = pokemon.replace(' ', '').replace('-', '').replace('.', '').lower()
        return self.pokemon[pokemon]

    def get_move(self, move=None):
        if move is None:
            return self.moves
        move = move.replace(' ', '').replace('-', '').replace('.', '').lower()
        moves = self.moves[move]
        moves['name'] = move
        moves['known'] = 0
        moves['used'] = 0
        return moves

    def get_games(self):
        return self.games

    def get_pokemon_learnset(self, pokemon_name: str) -> list:
        df = self.mp.get_movesets()
        filtered = df[df["pokemon"] == pokemon_name]
        moves = []
        if not filtered.empty:
            for i in range(1, 5):
                move = filtered[f'move_{i}'].iloc[0]
                if pd.isna(move):
                    moves.append({'name': 'None', 'accuracy': 0, 'base_power': 0, 'category': 'None', 'type': 'None', 'pp': 0, 'known': 0, 'used': 0})
                else:
                    move_stats = self.get_move(move)
                    moves.append(move_stats)
        else:
            for i in range(1, 5):
                moves.append({'name': 'None', 'accuracy': 0, 'base_power': 0, 'category': 'None', 'type': 'None', 'pp': 0, 'known': 0, 'used': 0})

        return moves



def main():
    '''
    Usage:
    - Getting Data
      dl = DataCollector(use_api=True, generation=5, format='ou', num_pages=100)

    - Loading Data (After Getting Data)
      dl = DataCollector(use_api=False, generation=5, format='ou')
      print(dl.get_move('Giga Drain'))
      print(dl.get_pokemon('Pikachu'))
      print(dl.get_pokemon_learnset('Tyranitar'))

    '''
    #dl = DataCollector(use_api=True, generation=5, format='ou', num_pages=100)
    dl = DataCollector(use_api=False, generation=5, format='ou')
    print(dl.get_move('Giga Drain'))
    print(dl.get_pokemon('Pikachu'))
    print(dl.get_pokemon_learnset('Aggron'))
    

if __name__ == "__main__":
    main()