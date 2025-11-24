import pypokedex
import pandas as pd
from web_scraper import DataLoader

GENERATION = 5
FORMAT = 'ou'

class GameParser():
    def __init__(self):
        self.data_loader = DataLoader(GENERATION, FORMAT)
        self.data_loader.get_move('CloseCombat')
        self.games = self.data_loader.get_game()
        self.meta = {}
        self.log = {}
        self.poke_stats = {}
        self.move_stats = {}
        
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
            if '|-message|' in line and ('forfeited' in line or 'lost due to inactivity' in line):
                # |-message|spindakaasie1 forfeited.
                return False

            elif '|gametype|' in line:
                game_info['gametype'] = line.split('|')[2]

            elif '|teamsize|' in line:
                if 'teamsize' not in game_info:
                    game_info['teamsize'] = {}
                game_info['teamsize'][line.split('|')[2]] = line.split('|')[3]

            elif '|poke|' in line:
                if 'poke' not in game_info:
                    game_info['poke'] = {}
                if line.split('|')[2] not in game_info['poke']:
                    game_info['poke'][line.split('|')[2]] = []
                pokemon = line.split('|')[3]
                game_info['poke'][line.split('|')[2]].append(pokemon)
                if ',' in pokemon:
                    pokemon_name = pokemon.split(',')[0]
                else:
                    pokemon_name = pokemon
                if pokemon_name not in self.poke_stats.keys():
                    self.poke_stats[pokemon_name] = self.data_loader.get_pokemon(pokemon_name)

            elif '|tier|' in line:
                game_info['tier'] = line.split('|')[2]

            elif '|switch|' in line:
                # |switch|p1a: Pretty|Gardevoir, M|100/100
                if 'start_poke' not in game_info:
                    game_info['start_poke'] = {}
                parts = line.split("|")
                player = parts[2].split(":")[0][:2]

                nickname = parts[2].split(': ')[1]
                pokemon = parts[3]
                if ',' in pokemon:
                    pokemon_name = pokemon.split(',')[0]
                else:
                    pokemon_name = pokemon
                health = parts[4]

                game_info['nickname_converter'][nickname] = pokemon_name

                game_info['start_poke'][player] = {
                    'species': pokemon_name,
                    'health': health
                }

            # determine winner slot (p1 or p2)
            elif '|win|' in line:
                winner_name = line.split('|')[2]
                if 'players' in game_info:
                    if game_info['players'].get('p1') == winner_name:
                        game_info['winner'] = 'p1'
                    elif game_info['players'].get('p2') == winner_name:
                        game_info['winner'] = 'p2'
                    else:
                        game_info['winner'] = None  # fail-safe

            if '|turn|' in line:
                break


        game_info['players'] = {}
        for line in game_text_split:
            # players
            if line.startswith('|player|') and line.count('|') > 3:
                parts = line.split('|')
                slot = parts[2]  # p1 or p2
                name = parts[3]
                game_info['players'][slot] = name

            if line.startswith('|win|'):
                winner_name = line.split('|')[2]
                if 'players' in game_info:
                    if game_info['players'].get('p1') == winner_name:
                        game_info['winner'] = 'p1'
                    elif game_info['players'].get('p2') == winner_name:
                        game_info['winner'] = 'p2'
                    else:
                        game_info['winner'] = None

        return True
    
    def _parse_game_data(self, game_id, game_text):
        response = self._parse_game_metadata(game_id, game_text)
        if response:
            self._parse_turns(game_id, game_text)
            current_poke = {'p1': self.meta[game_id]['start_poke']['p1']['species'], 'p2': self.meta[game_id]['start_poke']['p2']['species']}
            data = [] # {'col_name1': 'value1', 'col_name2': 'value2'}
            for turn_num, turn in self.log[game_id].items():
                faint = {'p1': False, 'p2': False}
                turn_dict = {
                    'game': game_id.split('-')[-1],
                    'turn': turn_num,
                    'active_p1': current_poke['p1'],
                    'active_p2': current_poke['p2'],
                    'move_p1': None,
                    'move_p2': None,
                    'switch_p1': False,
                    'switch_p2': False,
                    'weather': None,
                    #'hazards': None,
                    #'terrain': None
                }
                turn_num = int(turn_num)
                for event in turn:
                    if '|move|' in event:
                        # |move|p1a: persian|fake out|p2a: landorus
                        event_list = event.split('|')
                        player = event_list[2][:2]
                        move = event_list[3]
                        turn_dict[f'move_{player}'] = move

                        if move not in self.move_stats.keys():
                            self.move_stats[move] = self.data_loader.get_move(move)

                        nickname = event_list[2].split(': ')[1]
                        pokemon = self.meta[game_id]['nickname_converter'][nickname]
                        current_poke[player] = pokemon

                    elif '|-weather|' in event:
                        event_list = event.split('|')
                        weather = event_list[2]
                        turn_dict['weather'] = weather

                    elif '|faint|' in event:
                        event_list = event.split('|')
                        player = event_list[2][:2]
                        faint[player] = True


                    elif '|switch|' in event or '|drag|' in event or '|replace|' in event:
                        # '|switch|p2a: the moley spirit|excadrill, m|100/100',
                        # |replace|p2a: Zoroark|Zoroark, F
                        event_list = event.split('|')
                        player = event_list[2][:2]
                        if not faint[player]:
                            turn_dict[f'switch_{player}'] = True
                        
                        nickname = event_list[2].split(': ')[1]
                        pokemon = event_list[3]
                        if ',' in pokemon:
                            pokemon_name = pokemon.split(',')[0]
                        else:
                            pokemon_name = pokemon
                        if nickname not in self.meta[game_id]['nickname_converter'].keys():
                            self.meta[game_id]['nickname_converter'][nickname] = pokemon_name
                        current_poke[player] = pokemon_name
                        

                turn_dict['active_p1'] = current_poke['p1']
                turn_dict['active_p2'] = current_poke['p2']         

                data.append(turn_dict)

            df = pd.DataFrame(data)
            df["p1_win"] = 1 if self.meta[game_id]["winner"] == "p1" else 0
            return df
        
        return pd.DataFrame
    

    def parse_all_games(self):
        all_dfs = []

        for game_id, game_text in self.games.items():
            df = self._parse_game_data(game_id, game_text)
            if not df.empty:
                all_dfs.append(df)

        # Concatenate everything into one global DF
        full_df = pd.concat(all_dfs, ignore_index=True)
        return full_df
    

    def get_turn(self, game_id, turn_num):
        return self.log[game_id].get(turn_num)
    
    def get_meta(self, game_id):
        return self.meta[game_id]
    
    def get_poke_stats(self):
        return self.poke_stats
    
    def get_move_stats(self):
        return self.move_stats