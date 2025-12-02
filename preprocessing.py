from sklearn.preprocessing import StandardScaler

def preprocess_games(df):
    features = [x for x in df.columns if (x != 'p1_win')]

    # Get train, val, test games
    ids = df["game_id"].drop_duplicates()
    train_ids = ids[0:600]
    val_ids = ids[600:800]
    test_ids = ids[800:917]

    train_df = df[df["game_id"].isin(train_ids)]
    val_df = df[df["game_id"].isin(val_ids)]
    test_df = df[df["game_id"].isin(test_ids)]

    X_train = train_df[features]
    y_train = train_df["p1_win"]

    X_val = val_df[features]
    y_val = val_df["p1_win"]

    X_test = test_df[features]
    y_test = test_df["p1_win"]

    # Scale the train data, transform val and test data according to the scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
