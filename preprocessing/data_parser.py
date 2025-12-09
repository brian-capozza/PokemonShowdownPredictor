import pandas as pd
from DataCollector import DataCollector

ALL_TYPES = [
    "Normal","Fire","Water","Electric","Grass","Ice",
    "Fighting","Poison","Ground","Flying","Psychic",
    "Bug","Rock","Ghost","Dragon","Dark","Steel", "None"
]

WEATHER = [
    "Hail", "Sandstorm", "RainDaince", "SunnyDay", "None"
]

MOVE_CATEGORIES = [
    "Physical", "Special", "Status", "None"
]

STATUS_EFFECTS = [
    "tox", "brn", "par", "psn", "frz", "slp", "None"
]

DEFAULT_TURN_DICT = {
    'p1_switch': 0,
    'p2_switch': 0,
    'weather': 'None'
}

DEFAULT_POKE_BOOSTS = {
    'p1_hp_boost': 0,
    'p1_atk_boost': 0,
    'p1_def_boost': 0,
    'p1_spa_boost': 0,
    'p1_spd_boost': 0,
    'p1_spe_boost': 0,
    'p1_accuracy_boost': 0,
    'p1_evasion_boost': 0,

    'p2_hp_boost': 0,
    'p2_atk_boost': 0,
    'p2_def_boost': 0,
    'p2_spa_boost': 0,
    'p2_spd_boost': 0,
    'p2_spe_boost': 0,
    'p2_accuracy_boost': 0,
    'p2_evasion_boost': 0,
}

class GameParser():
    def __init__(self, generation, format):
        self.data_loader = DataCollector(use_api=False, generation=generation, format=format)
        self.meta = {}
        self.log = {}
        self.poke_stats = {}

    def _encode_slots(self, game_id, active_p1, active_p2):
        features = {'game_id': game_id.split('-')[-1]}

        p1_team = [p.split(",")[0] for p in self.meta[game_id]["poke"]["p1"]]
        p1_team = (p1_team + [None] * 6)[:6]


        p2_team = [p.split(",")[0] for p in self.meta[game_id]["poke"]["p2"]]
        p2_team = (p2_team + [None] * 6)[:6]

        def fill_slot(prefix, slot_index, species, active_species):
            slot = f"{prefix}_slot{slot_index}"

            game_info = self.meta[game_id]
            game_info['poke_to_slot'][prefix][species] = slot

            if species is not None:
                features[f"{slot}_active"] = 1 if species == active_species else 0

                stats = self.poke_stats[species]

                features[f"{slot}_current_health"] = self.meta[game_id]['poke_health'].get(f"{slot}_health", 100)
                features[f"{slot}_hp"]  = stats["hp"]
                features[f"{slot}_atk"] = stats["atk"]
                features[f"{slot}_def"] = stats["def"]
                features[f"{slot}_spa"] = stats["spa"]
                features[f"{slot}_spd"] = stats["spd"]
                features[f"{slot}_spe"] = stats["spe"]
                features[f"{slot}_status"] = self.meta[game_id]['poke_status'].get(f"{slot}_status", "None")

                t1 = stats["type1"]
                t2 = stats["type2"]

                for t in ALL_TYPES:
                    features[f"{slot}_type_{t}"] = int(t == t1 or t == t2)
            
            else:
                features[f"{slot}_active"] = 0

                features[f"{slot}_current_health"] = 0
                features[f"{slot}_hp"]  = 0
                features[f"{slot}_atk"] = 0
                features[f"{slot}_def"] = 0
                features[f"{slot}_spa"] = 0
                features[f"{slot}_spd"] = 0
                features[f"{slot}_spe"] = 0
                features[f"{slot}_status"] = "None"

                t1 = "None"
                t2 = "None"

                # One-hot types
                for t in ALL_TYPES:
                    features[f"{slot}_type_{t}"] = int(t == t1 or t == t2)


        for i, species in enumerate(p1_team, start=1):
            fill_slot("p1", i, species, active_p1)

        for i, species in enumerate(p2_team, start=1):
            fill_slot("p2", i, species, active_p2)

        return features


        
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
        game_info['nickname_converter']['p1'] = {}
        game_info['nickname_converter']['p2'] = {}
        game_info['poke_to_slot'] = {}
        game_info['poke_to_slot']['p1'] = {}
        game_info['poke_to_slot']['p2'] = {}
        game_info['poke_status'] = {}
        game_info['poke_health'] = {}
        game_info['poke_boosts'] = DEFAULT_POKE_BOOSTS.copy()
        game_info['poke_learnsets'] = {}
        game_info['poke_learnsets']['p1'] = {}
        game_info['poke_learnsets']['p2'] = {}
        game_text_split = game_text.split('\n')
        for line in game_text_split:
            if ('|-message|' in line and ('forfeited' in line or 'lost due to inactivity' in line)):
                # |-message|spindakaasie1 forfeited.
                return False
            
            elif '|rule|HP Percentage Mod: HP is reported as percentages' in line:
                # |rule|HP Percentage Mod: HP is reported as percentages
                # |rule|HP Percentage Mod: HP is shown in percentages
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

                poke_learnset = self.data_loader.get_pokemon_learnset(pokemon_name=pokemon_name)
                game_info['poke_learnsets'][line.split('|')[2]][pokemon_name] = poke_learnset

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

                game_info['nickname_converter'][player][nickname] = pokemon_name

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
        try:
            response = self._parse_game_metadata(game_id, game_text)
            if not response:
                return pd.DataFrame()

            self._parse_turns(game_id, game_text)

            # Current active pokémon (start of battle)
            current_poke = {
                'p1': self.meta[game_id]['start_poke']['p1']['species'],
                'p2': self.meta[game_id]['start_poke']['p2']['species']
            }

            data = []

            # Iterate through turns
            for turn_num, turn in self.log[game_id].items():
                faint = {'p1': 0, 'p2': 0}
                move_used = {'p1': (None, None), 'p2': (None, None)}
                turn_num = int(turn_num)

                # Build slot encoding (288D)
                slot_features = self._encode_slots(
                    game_id,
                    current_poke['p1'],
                    current_poke['p2']
                )

                turn_dict = DEFAULT_TURN_DICT.copy()

                for event in turn:
                    if '|-weather|' in event:
                        turn_dict['weather'] = event.split('|')[2]

                    elif '|faint|' in event:
                        event_list = event.split('|')
                        faint[player] = 1
                        player = event_list[2][:2]
                        nickname = event_list[2].split(': ')[1]
                        pokemon = self.meta[game_id]['nickname_converter'][player][nickname]
                        slot = self.meta[game_id]['poke_to_slot'][player][pokemon]
                        self.meta[game_id]['poke_health'][f"{slot}_health"] = 0

                    elif '|switch|' in event or '|drag|' in event or '|replace|' in event:
                        # |switch|p1a: Keldeo|Keldeo-Resolute|100/100
                        # |switch|p2a: Keldeo|Keldeo|100/100
                        event_list = event.split('|')
                        player = event_list[2][:2]

                        if not faint[player]:  
                            turn_dict[f"{player}_switch"] = 1

                        nickname = event_list[2].split(': ')[1]
                        pokemon_info = event_list[3]

                        pokemon_name = pokemon_info.split(',')[0]
                        self.meta[game_id]['nickname_converter'][player][nickname] = pokemon_name
                        current_poke[player] = pokemon_name

                        self.meta[game_id]['poke_boosts'] = DEFAULT_POKE_BOOSTS.copy()

                    elif '|move|' in event:
                        # |move|p1a: Persian|Fake Out|p2a: Landorus
                        event_list = event.split('|')
                        player = event_list[2][:2]  # "p1" or "p2"
                        move = event_list[3].replace(' ', '').replace('-', '').replace('.', '').lower()

                        nickname = event_list[2].split(': ')[1]
                        pokemon = self.meta[game_id]['nickname_converter'][player][nickname]
                        learnset = self.meta[game_id]['poke_learnsets'][player][pokemon]

                        move_used[player] = (pokemon, move)

                        # Load move stats
                        current_moves = []
                        for m in learnset:
                            current_moves.append(m['name'])
                        if move not in current_moves:
                            found_move = False
                            replace_index = None
                            for i in range(len(learnset), 0, -1):
                                if learnset[i-1]['known'] == 0 and learnset[i-1]['name'] != 'None':
                                    found_move = True
                                    learnset[i-1] = self.data_loader.get_move(move)
                                    learnset[i-1]['known'] = 1
                                    break
                                else:
                                    replace_index = i
                            if not found_move:
                                learnset[replace_index] = self.data_loader.get_move(move)
                                learnset[replace_index]['known'] = 1

                        # Update active pokemon based on nickname
                        nickname = event_list[2].split(': ')[1]
                        pokemon = self.meta[game_id]['nickname_converter'][player][nickname]
                        current_poke[player] = pokemon

                    elif '|-status|' in event:
                        event_list = event.split('|')
                        player = event_list[2][:2]
                        nickname = event_list[2].split(': ')[1]
                        pokemon = self.meta[game_id]['nickname_converter'][player][nickname]
                        slot = self.meta[game_id]['poke_to_slot'][player][pokemon]
                        self.meta[game_id]['poke_status'][f"{slot}_status"] = event_list[3]

                    elif '|-curestatus|' in event:
                        event_list = event.split('|')
                        player = event_list[2][:2]
                        nickname = event_list[2].split(': ')[1]
                        pokemon = self.meta[game_id]['nickname_converter'][player][nickname]
                        slot = self.meta[game_id]['poke_to_slot'][player][pokemon]
                        self.meta[game_id]['poke_status'][f"{slot}_status"] = 'None'

                    elif '|-unboost|' in event:
                        event_list = event.split('|')
                        player = event_list[2][:2]
                        nickname = event_list[2].split(': ')[1]
                        pokemon = self.meta[game_id]['nickname_converter'][player][nickname]
                        stat = event_list[3]
                        value = -int(event_list[4])

                        self.meta[game_id]['poke_boosts'][f'{player}_{stat}_boost'] += value

                    elif '|-boost|' in event:
                        event_list = event.split('|')
                        player = event_list[2][:2]
                        nickname = event_list[2].split(': ')[1]
                        pokemon = self.meta[game_id]['nickname_converter'][player][nickname]
                        stat = event_list[3]
                        value = int(event_list[4])

                        self.meta[game_id]['poke_boosts'][f'{player}_{stat}_boost'] += value

                    elif '|-damage|' in event or '|-heal|' in event:
                        event_list = event.split('|')
                        player = event_list[2][:2]
                        nickname = event_list[2].split(': ')[1]
                        pokemon = self.meta[game_id]['nickname_converter'][player][nickname]
                        slot = self.meta[game_id]['poke_to_slot'][player][pokemon]
                        if '/' not in event_list[3]:
                            health = 0
                        else:
                            health = int(event_list[3].split('/')[0])
                        self.meta[game_id]['poke_health'][f"{slot}_health"] = health
                        slot_features[f"{slot}_current_health"] = health

                    # |-unboost|p2a: Dragonite|atk|1
                    # |-boost|p2a: Dragonite|atk|1
                    # |-status|p2a: Politoed|tox
                    # |-curestatus|p2a: Politoed|tox|[msg]
                    # |-sidestart|p2: Birdmanmons|move: Toxic Spikes
                    # |-sideend|p2: Birdmanmons|move: Toxic Spikes|[of] p2a: Toxicroak
                    # |move|p1a: Jirachi|Ice Punch|p2a: Dragonite
                    # |-supereffective|p2a: Dragonite
                    # |-damage|p1a: Tyranitar|0 fnt
                    # |-damage|p2a: Garchomp|8/100
                    # |-heal|p1a: Rotom|100/100|[from] item: Leftovers


                for player in ['p1', 'p2']:
                    for pokemon, slot in self.meta[game_id]['poke_to_slot'][player].items():
                        learnset = self.meta[game_id]['poke_learnsets'][player][pokemon]
                        for i in range(len(learnset)):
                            move = learnset[i]
                            turn_dict[f'{slot}_move{i+1}_accuracy'] = move['accuracy']
                            turn_dict[f'{slot}_move{i+1}_base_power'] = move['base_power']
                            turn_dict[f'{slot}_move{i+1}_pp'] = move['pp']
                            if (
                                move_used[player][0] is not None
                                and pokemon == move_used[player][0]
                                and move['name'] == move_used[player][1]
                                and pokemon == current_poke[player]
                            ):
                                turn_dict[f'{slot}_move{i+1}_used'] = 1
                            else:
                                turn_dict[f'{slot}_move{i+1}_used'] = 0
                            # One-hot types
                            for t in ALL_TYPES:
                                turn_dict[f"{slot}_move{i+1}_type_{t}"] = int(t == move['type'])

                            # One-hot categories
                            for c in MOVE_CATEGORIES:
                                turn_dict[f"{slot}_move{i+1}_category_{c}"] = int(c == move['category'])

                # One-hot weather
                for w in WEATHER:
                    turn_dict[f"weather_{w}"] = int(w == turn_dict["weather"])

                turn_dict.pop("weather", None)

                # One-hot status
                keys_to_ohe = []
                for key in slot_features.keys():
                    if 'status' in key:
                        keys_to_ohe.append(key)

                for key in keys_to_ohe:
                    for s in STATUS_EFFECTS:
                        slot_features[f"{key}_{s}"] = int(s == slot_features[key])

                    slot_features.pop(key, None)
            

                row = {**slot_features, **turn_dict, **self.meta[game_id]['poke_boosts']}

                # DIMENSIONALITY CALC: Will always have this many columns
                # POKEMON: (((17 types + 7 stats + 6 status + 1 active/inactive)
                #   ((17 types + 3 categories + 3 numerical stats (acc, power, pp) + used/not used) * 4 moves)) * 12 pokemon = 1524
                # BOOSTS: (8 stats * 2 pokemon) = 16
                # WEATHER: 4 types
                # MISC: turn value + game id + who won + switch
                # TOTAL DIMS: 1524 + 16 + 4 + 4 = 1548

                row['turn'] = turn_num

                data.append(row)

            df = pd.DataFrame(data)

            df["p1_win"] = 1 if self.meta[game_id]["winner"] == "p1" else 0
            df["p1_win"] = 1 if self.meta[game_id]["winner"] == "p1" else 0
            return df
        except:
            print(f'Game {game_id} failed to parse...')
            return pd.DataFrame()

    

    def parse_all_games(self):
        all_dfs = []

        df = self.data_loader.get_games()

        for row in df.itertuples(): # type: ignore
            df = self._parse_game_data(row.game_id, row.raw_text)
            if not df.empty:
                all_dfs.append(df)

        # Concatenate everything into one global DF
        full_df = pd.concat(all_dfs, ignore_index=True)

        full_df = full_df.drop(columns=[col for col in full_df.columns if "None" in col])

        print(full_df.shape)

        return full_df