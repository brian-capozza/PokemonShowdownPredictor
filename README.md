# PokemonShowdownPredictor
## **Usage**

### **1. Preprocessing**
Run `preprocessing/DataCollector.py` to generate all required data files.  
This script produces:

- `gen5ou_game_text.pkl`
- `gen5ou_games.csv`
- `gen5ou_move_data.csv`
- `gen5ou_moves.json`
- `gen5ou_pokemon.json`

These files contain the processed battle logs, move metadata, and Pokémon information needed for model training.

---

### **2. Baselines**
Run `preprocessing/main.py` to process the baseline models on the generated data.

---

### **3. Model Training**
After preprocessing is complete, run `main_model.py`.  
All hyperparameters are defined in `globals.py`, including:

- **LSTM**: boolean - toggles between using an LSTM model or a feed-forward neural network.  
- **USE_FE**: boolean - enables or disables feature-engineered inputs.  
- Additional hyperparameters in `globals.py` may be adjusted as desired or left at default values.

---

## **AI Usage Statement**
ChatGPT was used to improve visualization plots, assist with code refactoring, and help implement features not covered in class. Model development and analysis were performed independently.

---
