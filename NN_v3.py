import tensorflow as tf
import numpy as np
import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
import time

start_time = time.time()

rolling = 5
epochs = 100
batch_size = 16
lr = 0.0001
nodes = 16
dropout = 0.3


# ---------------------------
# LOAD DATA
# ---------------------------
conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")
df = pd.read_sql("SELECT * FROM NN", conn)
conn.close()

# ---------------------------
# SORT CHRONOLOGICALLY
# ---------------------------
# df["game_id"] = df["game_id"].str.extract(r'(\d+)').astype(int)
# df = df.sort_values("game_id").reset_index(drop=True)

# ---------------------------
# BUILD EXPLICIT FOR/AGAINST
# ---------------------------
df["home_goals_for"] = df["home_goals"]
df["home_goals_against"] = df["away_goals"]

df["away_goals_for"] = df["away_goals"]
df["away_goals_against"] = df["home_goals"]

df["home_shots_for"] = df["home_shots"]
df["away_shots_for"] = df["away_shots"]

df["home_xg_for"] = df["home_xg"]
df["away_xg_for"] = df["away_xg"]

df["home_pim"] = df["home_penalty_minutes"]
df["away_pim"] = df["away_penalty_minutes"]


# ---------------------------
# HOME ROLLING FEATURES
# ---------------------------
df["home_last5_goals_for"] = (
    df.groupby("home_team")["home_goals_for"]
      .shift(1).rolling(rolling).mean()
)

df["home_last5_goals_against"] = (
    df.groupby("home_team")["home_goals_against"]
      .shift(1).rolling(rolling).mean()
)

df["home_last5_shots"] = (
    df.groupby("home_team")["home_shots_for"]
      .shift(1).rolling(rolling).mean()
)

df["home_last5_xg"] = (
    df.groupby("home_team")["home_xg_for"]
      .shift(1).rolling(rolling).mean()
)

df["home_last5_penalty_minutes"] = (
    df.groupby("home_team")["home_pim"]
      .shift(1).rolling(rolling).mean()
)

df["home_last5_winrate"] = (
    df.groupby("home_team")["home_win"]
      .shift(1).rolling(rolling).mean()
)

# ---------------------------
# AWAY ROLLING FEATURES
# ---------------------------
df["away_last5_goals_for"] = (
    df.groupby("away_team")["away_goals_for"]
      .shift(1).rolling(rolling).mean()
)

df["away_last5_goals_against"] = (
    df.groupby("away_team")["away_goals_against"]
      .shift(1).rolling(rolling).mean()
)

df["away_last5_shots"] = (
    df.groupby("away_team")["away_shots_for"]
      .shift(1).rolling(rolling).mean()
)

df["away_last5_xg"] = (
    df.groupby("away_team")["away_xg_for"]
      .shift(1).rolling(rolling).mean()
)

df["away_last5_penalty_minutes"] = (
    df.groupby("away_team")["away_pim"]
      .shift(1).rolling(rolling).mean()
)

df["away_last5_winrate"] = (
    df.groupby("away_team")["away_win"]
      .shift(1).rolling(rolling).mean()
)

# Create "Relative" features (Home vs Away)
df["rel_goals_for"] = df["home_last5_goals_for"] - df["away_last5_goals_for"]
df["rel_goals_against"] = df["home_last5_goals_against"] - df["away_last5_goals_against"]
df["rel_xg"] = df["home_last5_xg"] - df["away_last5_xg"]
df["rel_winrate"] = df["home_last5_winrate"] - df["away_last5_winrate"]
df["home_goals_dif"] = df["home_last5_goals_for"] - df["home_last5_goals_against"]
df["away_goals_dif"] = df["away_last5_goals_for"] - df["away_last5_goals_against"]

# ---------------------------
# DROP EARLY GAMES
# ---------------------------
df = df.dropna().reset_index(drop=True)

# ---------------------------
# TIME-BASED SPLIT
# ---------------------------
train = df.iloc[:int(0.6*len(df))]
valid = df.iloc[int(0.6*len(df)):int(0.8*len(df))]
test  = df.iloc[int(0.8*len(df)):]

# ---------------------------
# FEATURES
# ---------------------------
cat_cols = [
    "home_team",
    "away_team",

]

num_cols = [
    "home_last5_goals_for",
    "home_last5_goals_against",
    "home_last5_shots",
    "home_last5_xg",
    "home_last5_penalty_minutes",
    "home_last5_winrate",

    "away_last5_goals_for",
    "away_last5_goals_against",
    "away_last5_shots",
    "away_last5_xg",
    "away_last5_penalty_minutes",
    "away_last5_winrate",
    
    "rel_goals_for",
    "rel_goals_against",
    "rel_xg",
    "rel_winrate"
]

# ---------------------------
# PREPROCESSING
# ---------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

X_train_raw = train[cat_cols + num_cols]
y_train = train["home_win"]

X_valid_raw = valid[cat_cols + num_cols]
y_valid = valid["home_win"]

X_test_raw = test[cat_cols + num_cols]
y_test = test["home_win"]

preprocessor.fit(X_train_raw)

X_train = preprocessor.transform(X_train_raw).toarray()
X_valid = preprocessor.transform(X_valid_raw).toarray()
X_test  = preprocessor.transform(X_test_raw).toarray()

# ---------------------------
# BALANCE TRAINING DATA
# ---------------------------
ros = RandomOverSampler()
X_train, y_train = ros.fit_resample(X_train, y_train)

# ---------------------------
# MODEL
# ---------------------------
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(nodes, activation='relu', input_shape=(input_dim,), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

model = build_model(X_train.shape[1])

# early_stop = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=10,
#     restore_best_weights=True
# )

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=epochs,
    batch_size=batch_size,
    # callbacks=[early_stop],
    verbose=1
)

# ---------------------------
# EVALUATION
# ---------------------------
val_loss, val_acc = model.evaluate(X_valid, y_valid)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_acc)

y_pred = (model.predict(X_valid) > 0.5).astype(int).reshape(-1,)
print(classification_report(y_valid, y_pred))

# ---------------------------
# TEST SET
# ---------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

end_time = time.time()
print(f"Executed in: {end_time - start_time:.4f} seconds")
