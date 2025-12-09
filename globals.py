import torch

CSV_PATH = "datasets/gen5ou_games.csv"

LSTM = True
USE_FE = True

BATCH_SIZE = 64
EPOCHS = 60
LR = 8e-4
WEIGHT_DECAY = 1e-4

TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD_VALUE = -1.0

PATIENCE = 10

MIN_PREFIX_LEN = 3
PREFIX_MIN_FRAC_START = 0.3
PREFIX_MIN_FRAC_END = 0.3
current_prefix_min_frac = PREFIX_MIN_FRAC_START


GRAD_NOISE_STD = 0.003
LABEL_SMOOTHING = 0.05