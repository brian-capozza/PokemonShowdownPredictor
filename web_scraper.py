import requests

class DataLoader():
    def __init__(self, generation, format):
        self.generation = generation
        self.format = format
        self._build_games()
        self._build_moves()
        self._build_pokemon()


    def _build_games(self):
        self.games = {}
        base_url = 'https://replay.pokemonshowdown.com/'
        for page in range(1, 10):
            games_url = f'https://replay.pokemonshowdown.com/search.json?format=gen{self.generation}{self.format}&page={page}'
            data = requests.get(games_url).json()
            for game in data:
                game_data = requests.get(f'{base_url}{game["id"]}.log')
                self.games[game["id"]] = game_data.text
                #game_data = requests.get(f'{base_url}gen6ou-2484653408.log')
                #games['gen6ou-2484653408'] = game_data.text

    def _build_moves(self):
        self.moves = {}
        moves_url = "https://play.pokemonshowdown.com/data/moves.json"

        response = requests.get(moves_url)
        moves = response.json()

        for move, stats in moves.items():
            self.moves[move.lower()] = {
                'accuracy': stats['accuracy'],
                'base_power': stats['basePower'],
                'category': stats['category'],
                'type': stats['type'],
                'pp': stats['pp']
            }

    
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


    def get_pokemon(self, pokemon: str):
        filtered_pokemon = pokemon.replace(' ', '').replace('-', '').lower()
        if self.pokemon.get(filtered_pokemon) is None:
            return self.pokemon[pokemon.split('-')[0].lower()]
        return self.pokemon[filtered_pokemon]
    
    def get_move(self, move: str):
        return self.moves[move.replace(' ', '').replace('-', '').lower()]
    
    def get_game(self, game_id=None):
        if game_id is None:
            return self.games
        return self.games[game_id]










