import globals
from data import FeatureEngineering, SequenceBuilder, NNBuilder
from models import TinyLSTM, PokemonFullNet
from evaluation import ModelEvaluation

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

def main():
    print('Reading Data From CSV...')
    df = pd.read_csv(globals.CSV_PATH)
    fe = FeatureEngineering(df, use_fe=globals.USE_FE)
    X, y, game_ids, feature_cols = fe.run()


    if globals.LSTM:
        sb = SequenceBuilder(X, y, game_ids, feature_cols, globals.current_prefix_min_frac)
        ttvs = sb.train_test_val_split(globals.BATCH_SIZE, globals.TEST_SPLIT, globals.VAL_SPLIT, globals.MIN_PREFIX_LEN, globals.PAD_VALUE)
        (train_loader, val_loader, test_loader), \
        (game_test, test_seqs, test_labels), \
        y_train, \
        seq_builder = ttvs

        model = TinyLSTM(len(feature_cols)).to(globals.DEVICE)
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        optimizer = torch.optim.RMSprop(model.parameters(), lr=globals.LR, weight_decay=globals.WEIGHT_DECAY)

        me = ModelEvaluation(model, criterion, optimizer, globals.DEVICE)
        me.train_and_evaluate(
            train_loader,
            val_loader,
            test_loader,
            seq_builder,
            globals.EPOCHS,
            globals.PREFIX_MIN_FRAC_START,
            globals.PREFIX_MIN_FRAC_END,
            globals.LABEL_SMOOTHING,
            globals.PATIENCE,
            globals.GRAD_NOISE_STD,
        )

        me.visualize_training()
        me.visualize_games(game_test, test_seqs, test_labels)
        me.visualize_prf()

        importance = me.compute_feature_importance(
            test_loader,
            feature_names=feature_cols,
            num_batches=30,
        )
        me.plot_feature_importance(importance, feature_cols, top_k=40)
        me.summary()

    else:
        nb = NNBuilder(X, y, game_ids, feature_cols)
        train_loader, val_loader, test_loader, game_test, test_seqs, test_labels, scaler = \
            nb.train_test_val_split(globals.BATCH_SIZE, globals.TEST_SPLIT, globals.VAL_SPLIT)

        model = PokemonFullNet(input_layers=len(feature_cols)).to(globals.DEVICE)
        criterion = nn.BCEWithLogitsLoss(reduction="none")
        optimizer = torch.optim.Adam(model.parameters(), lr=globals.LR)

        me = ModelEvaluation(model, criterion, optimizer, globals.DEVICE)

        me.train_and_evaluate(
            train_loader,
            val_loader,
            test_loader,
            seq_builder=None,
            epochs=globals.EPOCHS,
            prefix_min_start=1.0,
            prefix_min_end=1.0,
            label_smoothing=0.0,
            patience=globals.PATIENCE,
            grad_noise_std=0.0,
        )

        # MLP visualization now works!
        me.visualize_training()
        me.visualize_games(game_test, test_seqs, test_labels)
        me.visualize_prf()

        importance = me.compute_feature_importance(
            test_loader,
            feature_names=feature_cols,
            num_batches=30,
        )

        me.plot_feature_importance(importance, feature_cols, top_k=40)
        me.summary()


if __name__ == '__main__':
    main()