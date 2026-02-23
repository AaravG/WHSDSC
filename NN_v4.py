import tensorflow as tf
import numpy as np
import pandas as pd
import sqlite3
from sklearn.preprocessing import RobustScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

# ---------------------------
# 1. THE DATA ENGINE
# ---------------------------
conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")
df = pd.read_sql("SELECT * FROM NN", conn)
conn.close()

# Sort games chronologically to respect the timeline
df["game_num"] = df["game_id"].str.extract(r'(\d+)').astype(int)
df = df.sort_values("game_num").reset_index(drop=True)

# Feature Engineering: Rolling xG (Last 10 games)
rolling = 10
for team in ['home', 'away']:
    df[f"{team}_roll_xg"] = df.groupby(f"{team}_team")[f"{team}_xg"].shift(1).rolling(rolling).mean()

# Calculate the "Signals" that actually showed impact in your tests
df["rel_elo"] = df["home_elo"] - df["away_elo"]
df["rel_xg"] = df["home_roll_xg"] - df["away_roll_xg"]

# Drop rows where rolling stats haven't accumulated yet
features = ["rel_elo", "rel_xg"]
df = df.dropna(subset=features + ["home_win"]).reset_index(drop=True)

# ---------------------------
# 2. TRAINING/TEST SPLIT
# ---------------------------
split = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split], df.iloc[split:]

X_train_raw = train_df[features]
y_train = train_df["home_win"]
X_test_raw = test_df[features]
y_test = test_df["home_win"]

# RobustScaler handles outliers better for sports data
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# ---------------------------
# 3. THE NEURAL NETWORK (V8 SNIPER)
# ---------------------------
# Compute weights to handle the Home/Away win imbalance
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_dict = dict(enumerate(weights))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(features),)),
    tf.keras.layers.Dense(12, activation='tanh'), # Tanh is better for relative (-/+) features
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(6, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Training Model...")
model.fit(
    X_train, y_train, 
    epochs=150, 
    batch_size=16, 
    class_weight=class_dict, 
    verbose=1
)

# ---------------------------
# 4. EVALUATION & SNIPER LOGIC
# ---------------------------
y_proba = model.predict(X_test).flatten()

# We only bet on the top 15% most confident Home Win predictions
home_threshold = np.percentile(y_proba, 85) 
mask = y_proba >= home_threshold

final_preds = (y_proba[mask] > 0.5).astype(int)
final_actuals = y_test.values[mask]

print("\n" + "="*40)
print("FINAL MODEL TRAINING COMPLETE")
print("="*40)
if len(final_actuals) > 0:
    acc = (final_preds == final_actuals).mean()
    print(f"HIGH-CONFIDENCE HOME ACCURACY: {acc:.2%}")
    print(f"TOTAL BETS IN TEST SET: {len(final_actuals)}")
    print("\nModel Status: Ready for Live Prediction")
else:
    print("Warning: Threshold too high, no bets found.")

# ---------------------------
# 5. LIVE PREDICTOR FUNCTION
# ---------------------------
def predict_new_matchup(h_elo, a_elo, h_xg_roll, a_xg_roll):
    """
    Input the current stats for a new game to see if it's a 'Sniper' bet.
    """
    rel_elo = h_elo - a_elo
    rel_xg = h_xg_roll - a_xg_roll
    
    scaled_input = scaler.transform([[rel_elo, rel_xg]])
    prob = model.predict(scaled_input, verbose=1)[0][0]
    
    print(f"\n--- Prediction ---")
    print(f"Home Win Probability: {prob:.2%}")
    if prob >= home_threshold:
        print("Action: 🎯 CONFIDENT HOME BET")
    else:
        print("Action: ❌ NO BET (Low Confidence)")

# Example call for a game tonight:
# predict_new_matchup(1550, 1480, 3.5, 2.8)