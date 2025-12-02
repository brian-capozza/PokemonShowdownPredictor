import pandas as pd
from web_scraper import DataLoader

GENERATION = 5
FORMAT = 'ou'

class GameParser():
    def __init__(self):
        self.data_loader = DataLoader(GENERATION, FORMAT)
        self.games = self.data_loader.get_game()
        self.meta = {}
        self.log = {}
        self.poke_stats = {}
        self.move_stats = {}

    def _encode_slots(self, game_id, active_p1, active_p2):
        """
        Builds a fixed-length one-hot encoded representation of both players' teams.
        Includes:
        - 6 slots per player
        - slotX_active = 1/0
        - stats for each Pokémon
        """

        features = {'game_id': game_id.split('-')[-1]}

        # fill to be length six
        p1_team = [p.split(",")[0] for p in self.meta[game_id]["poke"]["p1"]]
        # Pad or trim to length 6
        p1_team = (p1_team + [None] * 6)[:6]


        p2_team = [p.split(",")[0] for p in self.meta[game_id]["poke"]["p2"]]
        # Pad or trim to length 6
        p2_team = (p2_team + [None] * 6)[:6]

        # -----------------------------
        # Helper for filling slot stats
        # -----------------------------
        def fill_slot(prefix, slot_index, species, active_species):
            """
            prefix: "p1" or "p2"
            slot_index: 1..6
            species: Pokémon name in this slot
            active_species: current active Pokémon
            """
            slot = f"{prefix}_slot{slot_index}"

            # Standard type list for one-hot
            all_types = [
                "Normal","Fire","Water","Electric","Grass","Ice",
                "Fighting","Poison","Ground","Flying","Psychic",
                "Bug","Rock","Ghost","Dragon","Dark","Steel", "None"
            ]

            if species is not None:
                # Active flag
                features[f"{slot}_active"] = 1 if species == active_species else 0

                # Retrieve stats
                stats = self.poke_stats[species]

                features[f"{slot}_hp"]  = stats["hp"]
                features[f"{slot}_atk"] = stats["atk"]
                features[f"{slot}_def"] = stats["def"]
                features[f"{slot}_spa"] = stats["spa"]
                features[f"{slot}_spd"] = stats["spd"]
                features[f"{slot}_spe"] = stats["spe"]

                t1 = stats["type1"]
                t2 = stats["type2"]

                # One-hot types
                for t in all_types:
                    features[f"{slot}_type_{t}"] = int(t == t1 or t == t2)
            
            else:
                # Active flag
                features[f"{slot}_active"] = 0

                features[f"{slot}_hp"]  = 0
                features[f"{slot}_atk"] = 0
                features[f"{slot}_def"] = 0
                features[f"{slot}_spa"] = 0
                features[f"{slot}_spd"] = 0
                features[f"{slot}_spe"] = 0

                t1 = "None"
                t2 = "None"

                # One-hot types
                for t in all_types:
                    features[f"{slot}_type_{t}"] = int(t == t1 or t == t2)


        # -----------------------------
        # Fill all 6 slots for p1
        # -----------------------------
        for i, species in enumerate(p1_team, start=1):
            fill_slot("p1", i, species, active_p1)

        # -----------------------------
        # Fill all 6 slots for p2
        # -----------------------------
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
        game_text_split = game_text.split('\n')
        for line in game_text_split:
            if ('|-message|' in line and ('forfeited' in line or 'lost due to inactivity' in line)) or ('|raw|' in line):
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
            turn_num = int(turn_num)

            # ---------------------------
            # Create turn-specific fields
            # ----------------------------
            turn_dict = {
                'p1_move_accuracy': 0,
                'p1_move_basepower': 0,
                'p1_move_pp': 0,
                'p1_move_category': 'Other',
                'p1_move_type': 'Other',
                'p1_switch': 0,

                'p2_move_accuracy': 0,
                'p2_move_basepower': 0,
                'p2_move_pp': 0,
                'p2_move_category': 'Other',
                'p2_move_type': 'Other',
                'p2_switch': 0,

                'weather': 'None'
            }

            # ---------------------------
            # Parse each event in turn
            # ---------------------------
            for event in turn:

                # ─────────── MOVE ───────────
                if '|move|' in event:
                    # |move|p1a: Persian|Fake Out|p2a: Landorus
                    event_list = event.split('|')
                    player = event_list[2][:2]  # "p1" or "p2"
                    move = event_list[3]

                    # Load move stats if missing
                    if move not in self.move_stats:
                        self.move_stats[move] = self.data_loader.get_move(move)

                    m = self.move_stats[move]
                    if m.get("accuracy"):
                        turn_dict[f"{player}_move_accuracy"] = 100
                    else:
                        turn_dict[f"{player}_move_accuracy"] = m.get("accuracy")
                    turn_dict[f"{player}_move_basepower"]  = m.get("base_power")
                    turn_dict[f"{player}_move_pp"]         = m.get("pp")
                    turn_dict[f"{player}_move_category"]   = m.get("category")
                    turn_dict[f"{player}_move_type"]       = m.get("type")

                    # Update active pokemon based on nickname
                    nickname = event_list[2].split(': ')[1]
                    pokemon = self.meta[game_id]['nickname_converter'][nickname]
                    current_poke[player] = pokemon

                # ─────────── WEATHER ───────────
                elif '|-weather|' in event:
                    turn_dict['weather'] = event.split('|')[2]

                # ─────────── FAINT ───────────
                elif '|faint|' in event:
                    player = event.split('|')[2][:2]
                    faint[player] = 1

                # ─────────── SWITCH / DRAG / REPLACE ───────────
                elif '|switch|' in event or '|drag|' in event or '|replace|' in event:
                    event_list = event.split('|')
                    player = event_list[2][:2]

                    if not faint[player]:  
                        turn_dict[f"{player}_switch"] = 1

                    nickname = event_list[2].split(': ')[1]
                    pokemon_info = event_list[3]

                    pokemon_name = pokemon_info.split(',')[0]
                    self.meta[game_id]['nickname_converter'][nickname] = pokemon_name
                    current_poke[player] = pokemon_name

            # ---------------------------
            # Build slot encoding (288D)
            # ---------------------------
            slot_features = self._encode_slots(
                game_id,
                current_poke['p1'],
                current_poke['p2']
            )

            # ---------------------------
            # Final row = slot features + turn metadata
            # ---------------------------
            row = {**slot_features, **turn_dict}

            # DIMENSIONALITY CALC:
            # POKEMON: ((17 types + 6 stats + 1 active/inactive) * 12 pokemon) = 288
            # MOVES: ((17 types + 2 categories + 4 numerical stats (acc, power, pp, switch)) * 2 moves) = 46
            # WEATHER: 4 types
            # MISC: turn value + game id + who won
            # TOTAL DIMS: 288 + 46 + 4 + 3 = 341

            row['turn'] = turn_num

            data.append(row)

        df = pd.DataFrame(data)

        df["p1_win"] = 1 if self.meta[game_id]["winner"] == "p1" else 0
        df["p1_win"] = 1 if self.meta[game_id]["winner"] == "p1" else 0
        return df

    

    def parse_all_games(self):
        all_dfs = []

        for game_id, game_text in self.games.items():
            df = self._parse_game_data(game_id, game_text)
            if not df.empty:
                all_dfs.append(df)

        # Concatenate everything into one global DF
        full_df = pd.concat(all_dfs, ignore_index=True)

        full_df = pd.get_dummies(
            full_df,
            columns=[
                "p1_move_type", "p2_move_type",
                "p1_move_category", "p2_move_category",
                "weather"
            ],
            prefix_sep="_",
            dtype=int  # forces 0/1 instead of True/False
        )

        print(full_df.shape)

        return full_df
    

    def get_turn(self, game_id, turn_num):
        return self.log[game_id].get(turn_num)
    
    def get_meta(self, game_id):
        return self.meta[game_id]
    
    def get_poke_stats(self):
        return self.poke_stats
    
    def get_move_stats(self):
        return self.move_stats