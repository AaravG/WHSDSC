import tensorflow as tf
import numpy as np
import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import time

start_time = time.time()

# ---------------------------
# HYPERPARAMETERS
# ---------------------------
rolling = 10
epochs = 100
batch_size = 32
lr = 0.001
dropout = 0.3

print("="*60)
print("FIXED NEURAL NETWORK MODEL")
print("="*60)

# ---------------------------
# LOAD DATA
# ---------------------------
conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")
df = pd.read_sql("SELECT * FROM NN", conn)
conn.close()

print(f"✓ Loaded {len(df)} games")

# ---------------------------
# CRITICAL FIX: Sort by game_id (chronological order)
# ---------------------------
# Extract numeric part from game_id
df["game_num"] = df["game_id"].str.extract(r'(\d+)').astype(int)
df = df.sort_values("game_num").reset_index(drop=True)
print(f"✓ Sorted by game_id (chronological order)")

# Verify sort worked
print(f"\nFirst 5 games:")
print(df[['game_id', 'home_team', 'away_team']].head())
print(f"\nLast 5 games:")
print(df[['game_id', 'home_team', 'away_team']].tail())

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

if "away_win" not in df.columns:
    df["away_win"] = 1 - df["home_win"]

# ---------------------------
# ELO FEATURES
# ---------------------------
df['elo_diff'] = df['home_elo'] - df['away_elo']

# Check ELO correlation (should improve after sorting)
elo_corr = np.corrcoef(df['elo_diff'], df['home_win'])[0, 1]
print(f"\n✓ ELO correlation with home_win: {elo_corr:+.4f}")

if abs(elo_corr) < 0.10:
    print("  ⚠️  ELO correlation is weak - but we'll use it anyway")
else:
    print("  ✓ ELO has predictive power")

# ---------------------------
# HOME ROLLING FEATURES
# ---------------------------
print("\nCalculating rolling features (chronologically)...")

df["home_last5_goals_for"] = (
    df.groupby("home_team")["home_goals_for"]
      .shift(1).rolling(rolling, min_periods=1).mean()
)

df["home_last5_goals_against"] = (
    df.groupby("home_team")["home_goals_against"]
      .shift(1).rolling(rolling, min_periods=1).mean()
)

df["home_last5_goal_diff"] = df["home_last5_goals_for"] - df["home_last5_goals_against"]

df["home_last5_shots"] = (
    df.groupby("home_team")["home_shots_for"]
      .shift(1).rolling(rolling, min_periods=1).mean()
)

df["home_last5_xg"] = (
    df.groupby("home_team")["home_xg_for"]
      .shift(1).rolling(rolling, min_periods=1).mean()
)

df["home_last5_shooting_pct"] = (
    df.groupby("home_team")["home_goals_for"]
      .shift(1).rolling(rolling, min_periods=1).sum()
) / (
    df.groupby("home_team")["home_shots_for"]
      .shift(1).rolling(rolling, min_periods=1).sum()
)

df["home_last5_penalty_minutes"] = (
    df.groupby("home_team")["home_pim"]
      .shift(1).rolling(rolling, min_periods=1).mean()
)

df["home_last5_winrate"] = (
    df.groupby("home_team")["home_win"]
      .shift(1).rolling(rolling, min_periods=1).mean()
)

df["home_last3_winrate"] = (
    df.groupby("home_team")["home_win"]
      .shift(1).rolling(3, min_periods=1).mean()
)

# ---------------------------
# AWAY ROLLING FEATURES
# ---------------------------
df["away_last5_goals_for"] = (
    df.groupby("away_team")["away_goals_for"]
      .shift(1).rolling(rolling, min_periods=1).mean()
)

df["away_last5_goals_against"] = (
    df.groupby("away_team")["away_goals_against"]
      .shift(1).rolling(rolling, min_periods=1).mean()
)

df["away_last5_goal_diff"] = df["away_last5_goals_for"] - df["away_last5_goals_against"]

df["away_last5_shots"] = (
    df.groupby("away_team")["away_shots_for"]
      .shift(1).rolling(rolling, min_periods=1).mean()
)

df["away_last5_xg"] = (
    df.groupby("away_team")["away_xg_for"]
      .shift(1).rolling(rolling, min_periods=1).mean()
)

df["away_last5_shooting_pct"] = (
    df.groupby("away_team")["away_goals_for"]
      .shift(1).rolling(rolling, min_periods=1).sum()
) / (
    df.groupby("away_team")["away_shots_for"]
      .shift(1).rolling(rolling, min_periods=1).sum()
)

df["away_last5_penalty_minutes"] = (
    df.groupby("away_team")["away_pim"]
      .shift(1).rolling(rolling, min_periods=1).mean()
)

df["away_last5_winrate"] = (
    df.groupby("away_team")["away_win"]
      .shift(1).rolling(rolling, min_periods=1).mean()
)

df["away_last3_winrate"] = (
    df.groupby("away_team")["away_win"]
      .shift(1).rolling(3, min_periods=1).mean()
)

# ---------------------------
# RELATIVE FEATURES
# ---------------------------
df["rel_goals_for"] = df["home_last5_goals_for"] - df["away_last5_goals_for"]
df["rel_goals_against"] = df["home_last5_goals_against"] - df["away_last5_goals_against"]
df["rel_goal_diff"] = df["home_last5_goal_diff"] - df["away_last5_goal_diff"]
df["rel_xg"] = df["home_last5_xg"] - df["away_last5_xg"]
df["rel_shooting_pct"] = df["home_last5_shooting_pct"] - df["away_last5_shooting_pct"]
df["rel_winrate"] = df["home_last5_winrate"] - df["away_last5_winrate"]
df["rel_momentum"] = df["home_last3_winrate"] - df["away_last3_winrate"]
# The "Power Index"
df['power_index'] = df['rel_xg'] * df['rel_winrate']



# ---------------------------
# CLEAN DATA
# ---------------------------
df = df.replace([np.inf, -np.inf], np.nan)
initial_rows = len(df)
df = df.dropna().reset_index(drop=True)
print(f"✓ Dropped {initial_rows - len(df)} rows with missing data")
print(f"✓ Final dataset: {len(df)} games")

# ---------------------------
# TIME-BASED SPLIT (chronological)
# ---------------------------
train = df.iloc[:int(0.6*len(df))]
valid = df.iloc[int(0.6*len(df)):int(0.8*len(df))]
test  = df.iloc[int(0.8*len(df)):]

print(f"\nDataset split (chronological):")
print(f"  Train: {len(train)} games")
print(f"  Valid: {len(valid)} games")
print(f"  Test:  {len(test)} games")

# ---------------------------
# FEATURES
# ---------------------------
# cat_cols = ["home_team", "away_team"]

num_cols = [
    "elo_diff",       # The difference matters more than the raw scores
    "rel_goal_diff", 
    "rel_momentum",
    "rel_shooting_pct",
    "power_index"     # This now carries the weight of xG and Winrate
]

# print(f"\nFeatures: {len(cat_cols)} categorical, {len(num_cols)} numerical")

# ---------------------------
# PREPROCESSING
# ---------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    # ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# X_train_raw = train[cat_cols + num_cols]
# y_train = train["home_win"]

# X_valid_raw = valid[cat_cols + num_cols]
# y_valid = valid["home_win"]

# X_test_raw = test[cat_cols + num_cols]
# y_test = test["home_win"]

X_train_raw = train[num_cols]
y_train = train["home_win"]

X_valid_raw = valid[num_cols]
y_valid = valid["home_win"]

X_test_raw = test[num_cols]
y_test = test["home_win"]

preprocessor.fit(X_train_raw)

X_train_transformed = preprocessor.transform(X_train_raw)
X_valid_transformed = preprocessor.transform(X_valid_raw)
X_test_transformed = preprocessor.transform(X_test_raw)

# Handle sparse/dense conversion
if hasattr(X_train_transformed, 'toarray'):
    X_train = X_train_transformed.toarray()
    X_valid = X_valid_transformed.toarray()
    X_test = X_test_transformed.toarray()
else:
    X_train = X_train_transformed
    X_valid = X_valid_transformed
    X_test = X_test_transformed

print(f"✓ Preprocessed feature dimension: {X_train.shape[1]}")

# ---------------------------
# BALANCE TRAINING DATA
# ---------------------------
ros = RandomOverSampler(random_state=42)
X_train, y_train = ros.fit_resample(X_train, y_train)
print(f"✓ Balanced training set: {len(y_train)} samples")

# ---------------------------
# MODEL
# ---------------------------
def build_model(input_dim):
    model = tf.keras.Sequential([
        # Layer 1: Wider to catch signals, use leaky_relu to keep neurons alive
        tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01), input_dim=input_dim),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        # Layer 2: Transition layer
        tf.keras.layers.Dense(32),
        tf.keras.layers.LeakyReLU(alpha=0.1),
        tf.keras.layers.Dropout(0.3),
        
        # Output
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), # Slightly higher than 0.0003
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

model = build_model(X_train.shape[1])

# ---------------------------
# CALLBACKS
# ---------------------------
patience = 30  # Give it significantly more room to learn

# --- UPDATE CALLBACKS ---
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=patience,           # Increase this!
    restore_best_weights=True,
    verbose=1
)

# 2. Add a scheduler to the callbacks
lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.75,     # Reduce LR by 25% when plateauing
    patience=5,     # Wait 5 epochs
    min_lr=0.00001
)

# ---------------------------
# TRAIN
# ---------------------------
print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

# Calculate weights to balance the 'Home Bias'
total = len(y_train)
pos = np.sum(y_train) # Home wins
neg = total - pos    # Away wins

# Higher weight for the minority class (usually Away wins)
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

# Then add this to your model.fit:
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=1000,
    batch_size=32, # Increase batch size slightly for smoother gradients
    class_weight=class_weight,
    callbacks=[early_stop, lr_reducer], # Added lr_reducer here
    verbose=1
)

# ---------------------------
# VALIDATION EVALUATION
# ---------------------------
print("\n" + "="*60)
print("VALIDATION SET PERFORMANCE")
print("="*60)

val_loss, val_acc, val_auc = model.evaluate(X_valid, y_valid, verbose=0)
print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print(f"Validation AUC:      {val_auc:.4f}")

y_pred_proba = model.predict(X_valid, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).reshape(-1,)

print("\n" + classification_report(y_valid, y_pred, target_names=["Away Win", "Home Win"]))

# ---------------------------
# TEST EVALUATION
# ---------------------------
print("\n" + "="*60)
print("TEST SET PERFORMANCE (FINAL)")
print("="*60)

test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test AUC:      {test_auc:.4f}")

y_test_pred_proba = model.predict(X_test, verbose=0)
y_test_pred = (y_test_pred_proba > 0.5).astype(int).reshape(-1,)

print("\n" + classification_report(y_test, y_test_pred, target_names=["Away Win", "Home Win"]))

# ---------------------------
# SUMMARY
# ---------------------------
end_time = time.time()

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Training time:       {end_time - start_time:.2f} seconds")
print(f"Validation Accuracy: {val_acc:.2%}")
print(f"Test Accuracy:       {test_acc:.2%}")
print(f"Test AUC:            {test_auc:.4f}")

if test_acc >= 0.65:
    print("\n✓ Model is working! Accuracy ≥ 65%")
elif test_acc >= 0.55:
    print("\n⚠️  Model is weak but better than random (55-65%)")
else:
    print("\n❌ Model still not working (<55%)")
    print("   Additional debugging needed")

# Save model
model.save('hockey_predictor_fixed.h5')
print("\n✓ Model saved to 'hockey_predictor_fixed.h5'")