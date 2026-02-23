import pandas as pd
import numpy as np
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import time

start_time = time.time()

print("="*60)
print("LOGISTIC REGRESSION MODEL (Production)")
print("="*60)

# ==========================
# 1. LOAD DATA
# ==========================
conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")
df = pd.read_sql("SELECT * FROM NN", conn)
conn.close()

print(f"✓ Loaded {len(df)} games")

# ==========================
# 2. SORT CHRONOLOGICALLY
# ==========================
df["game_num"] = df["game_id"].str.extract(r'(\d+)').astype(int)
df = df.sort_values("game_num").reset_index(drop=True)
print(f"✓ Sorted chronologically")

# ==========================
# 3. FEATURE ENGINEERING
# ==========================
print("\nCalculating features...")

rolling = 10

# Create for/against columns
df["home_goals_for"] = df["home_goals"]
df["home_goals_against"] = df["away_goals"]
df["away_goals_for"] = df["away_goals"]
df["away_goals_against"] = df["home_goals"]

df["home_shots_for"] = df["home_shots"]
df["away_shots_for"] = df["away_shots"]

df["home_xg_for"] = df["home_xg"]
df["away_xg_for"] = df["away_xg"]

if "away_win" not in df.columns:
    df["away_win"] = 1 - df["home_win"]

# === HOME TEAM ROLLING STATS ===
df["home_last10_goals_for"] = (
    df.groupby("home_team")["home_goals_for"]
      .shift(1).rolling(rolling, min_periods=3).mean()
)

df["home_last10_goals_against"] = (
    df.groupby("home_team")["home_goals_against"]
      .shift(1).rolling(rolling, min_periods=3).mean()
)

df["home_last10_goal_diff"] = df["home_last10_goals_for"] - df["home_last10_goals_against"]

df["home_last10_shots"] = (
    df.groupby("home_team")["home_shots_for"]
      .shift(1).rolling(rolling, min_periods=3).mean()
)

df["home_last10_xg"] = (
    df.groupby("home_team")["home_xg_for"]
      .shift(1).rolling(rolling, min_periods=3).mean()
)

df["home_last10_winrate"] = (
    df.groupby("home_team")["home_win"]
      .shift(1).rolling(rolling, min_periods=3).mean()
)

# Recent momentum (last 3 games)
df["home_last3_winrate"] = (
    df.groupby("home_team")["home_win"]
      .shift(1).rolling(3, min_periods=1).mean()
)

# === AWAY TEAM ROLLING STATS ===
df["away_last10_goals_for"] = (
    df.groupby("away_team")["away_goals_for"]
      .shift(1).rolling(rolling, min_periods=3).mean()
)

df["away_last10_goals_against"] = (
    df.groupby("away_team")["away_goals_against"]
      .shift(1).rolling(rolling, min_periods=3).mean()
)

df["away_last10_goal_diff"] = df["away_last10_goals_for"] - df["away_last10_goals_against"]

df["away_last10_shots"] = (
    df.groupby("away_team")["away_shots_for"]
      .shift(1).rolling(rolling, min_periods=3).mean()
)

df["away_last10_xg"] = (
    df.groupby("away_team")["away_xg_for"]
      .shift(1).rolling(rolling, min_periods=3).mean()
)

df["away_last10_winrate"] = (
    df.groupby("away_team")["away_win"]
      .shift(1).rolling(rolling, min_periods=3).mean()
)

df["away_last3_winrate"] = (
    df.groupby("away_team")["away_win"]
      .shift(1).rolling(3, min_periods=1).mean()
)

# === RELATIVE FEATURES (MOST IMPORTANT) ===
df["elo_diff"] = df["home_elo"] - df["away_elo"]
df["rel_goal_diff"] = df["home_last10_goal_diff"] - df["away_last10_goal_diff"]
df["rel_winrate"] = df["home_last10_winrate"] - df["away_last10_winrate"]
df["rel_momentum"] = df["home_last3_winrate"] - df["away_last3_winrate"]
df["rel_xg"] = df["home_last10_xg"] - df["away_last10_xg"]
df["rel_shots"] = df["home_last10_shots"] - df["away_last10_shots"]

# Home ice advantage constant
df["home_advantage"] = 1.0

# ==========================
# 4. CLEAN DATA
# ==========================
features = [
    "elo_diff",
    "rel_goal_diff",
    "rel_winrate",
    "rel_momentum",
    "rel_xg",
    "rel_shots",
    "home_advantage"
]

df = df.replace([np.inf, -np.inf], np.nan)
df_clean = df.dropna(subset=features + ["home_win"]).reset_index(drop=True)

print(f"✓ Final dataset: {len(df_clean)} games")

# ==========================
# 5. TRAIN/VALID/TEST SPLIT
# ==========================
train = df_clean.iloc[:int(0.6*len(df_clean))]
valid = df_clean.iloc[int(0.6*len(df_clean)):int(0.8*len(df_clean))]
test = df_clean.iloc[int(0.8*len(df_clean)):]

print(f"\nSplit: Train={len(train)}, Valid={len(valid)}, Test={len(test)}")

X_train = train[features]
y_train = train["home_win"]

X_valid = valid[features]
y_valid = valid["home_win"]

X_test = test[features]
y_test = test["home_win"]

# ==========================
# 6. SCALE FEATURES
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# ==========================
# 7. TRAIN LOGISTIC REGRESSION
# ==========================
print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

model = LogisticRegression(
    class_weight='balanced',  # Handles home/away win imbalance
    max_iter=1000,
    random_state=42,
    C=1.0  # Regularization strength (lower = more regularization)
)

model.fit(X_train_scaled, y_train)

print("✓ Model trained successfully")

# ==========================
# 8. FEATURE IMPORTANCE
# ==========================
print("\n" + "="*60)
print("FEATURE IMPORTANCE (Coefficients)")
print("="*60)

coefficients = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_[0],
    'abs_coefficient': np.abs(model.coef_[0])
}).sort_values('abs_coefficient', ascending=False)

for _, row in coefficients.iterrows():
    print(f"{row['feature']:20s}: {row['coefficient']:+.4f}")

# ==========================
# 9. VALIDATION PERFORMANCE
# ==========================
print("\n" + "="*60)
print("VALIDATION SET PERFORMANCE")
print("="*60)

y_valid_pred = model.predict(X_valid_scaled)
y_valid_proba = model.predict_proba(X_valid_scaled)[:, 1]

valid_acc = (y_valid_pred == y_valid).mean()
valid_auc = roc_auc_score(y_valid, y_valid_proba)

print(f"Accuracy: {valid_acc:.4f} ({valid_acc*100:.2f}%)")
print(f"AUC:      {valid_auc:.4f}")

print("\n" + classification_report(y_valid, y_valid_pred, target_names=["Away Win", "Home Win"]))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_valid, y_valid_pred)
print(f"              Predicted")
print(f"              Away  Home")
print(f"Actual Away   {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"       Home   {cm[1,0]:4d}  {cm[1,1]:4d}")

# ==========================
# 10. TEST PERFORMANCE
# ==========================
print("\n" + "="*60)
print("TEST SET PERFORMANCE (FINAL)")
print("="*60)

y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

test_acc = (y_test_pred == y_test).mean()
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"AUC:      {test_auc:.4f}")

print("\n" + classification_report(y_test, y_test_pred, target_names=["Away Win", "Home Win"]))

print(f"\nHome win rate: {y_test.mean():.1%}")

# ==========================
# 11. CONFIDENCE-BASED FILTERING
# ==========================
print("\n" + "="*60)
print("CONFIDENCE FILTERING ANALYSIS")
print("="*60)

# Test different confidence thresholds
thresholds = [0.60, 0.65, 0.70, 0.75]

for threshold in thresholds:
    # Filter for confident predictions
    confident_mask = (y_test_proba >= threshold) | (y_test_proba <= (1 - threshold))
    
    if confident_mask.sum() > 0:
        y_test_confident = y_test[confident_mask]
        y_pred_confident = y_test_pred[confident_mask]
        
        conf_acc = (y_pred_confident == y_test_confident).mean()
        n_bets = confident_mask.sum()
        
        print(f"\nThreshold {threshold:.2f}:")
        print(f"  Accuracy: {conf_acc:.2%}")
        print(f"  # of bets: {n_bets} ({n_bets/len(y_test)*100:.1f}% of games)")
        print(f"  Expected ROI: {(conf_acc - 0.525) * 100:.1f}%")  # 52.5% = break-even

# ==========================
# 12. VISUALIZATIONS
# ==========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
axes[0].plot(fpr, tpr, label=f'ROC Curve (AUC = {test_auc:.3f})', linewidth=2)
axes[0].plot([0, 1], [0, 1], 'k--', label='Random Guess')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Probability Distribution
axes[1].hist(y_test_proba[y_test == 0], bins=20, alpha=0.5, label='Away Wins', color='red')
axes[1].hist(y_test_proba[y_test == 1], bins=20, alpha=0.5, label='Home Wins', color='blue')
axes[1].axvline(0.5, color='black', linestyle='--', label='Decision Boundary')
axes[1].set_xlabel('Predicted Probability (Home Win)')
axes[1].set_ylabel('Count')
axes[1].set_title('Prediction Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_regression_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved to 'logistic_regression_analysis.png'")

# ==========================
# 13. CALIBRATION ANALYSIS
# ==========================
print("\n" + "="*60)
print("CALIBRATION ANALYSIS")
print("="*60)

bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2

digitized = np.digitize(y_test_proba, bins) - 1
bin_accuracies = []
bin_counts = []

for i in range(len(bin_centers)):
    mask = digitized == i
    if mask.sum() > 0:
        bin_acc = y_test[mask].mean()
        bin_accuracies.append(bin_acc)
        bin_counts.append(mask.sum())
    else:
        bin_accuracies.append(np.nan)
        bin_counts.append(0)

print("\nPredicted Prob | Actual Win Rate | # Games")
print("-" * 45)
for center, acc, count in zip(bin_centers, bin_accuracies, bin_counts):
    if count > 0:
        print(f"    {center:.2f}      |      {acc:.2f}      |   {count:3d}")

# ==========================
# 14. SAVE MODEL
# ==========================
import pickle

model_data = {
    'model': model,
    'scaler': scaler,
    'features': features,
    'test_accuracy': test_acc,
    'test_auc': test_auc
}

with open('logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("\n✓ Model saved to 'logistic_regression_model.pkl'")

# ==========================
# 15. PREDICTION FUNCTION
# ==========================
def predict_new_game(home_team_stats, away_team_stats):
    """
    Predict a new game.
    
    Parameters:
    -----------
    home_team_stats : dict
        {
            'elo': 1550,
            'last10_goal_diff': 0.5,
            'last10_winrate': 0.6,
            'last3_winrate': 0.67,
            'last10_xg': 3.2,
            'last10_shots': 28.5
        }
    away_team_stats : dict (same structure)
    
    Returns:
    --------
    dict with prediction results
    """
    # Calculate relative features
    input_data = {
        'elo_diff': home_team_stats['elo'] - away_team_stats['elo'],
        'rel_goal_diff': home_team_stats['last10_goal_diff'] - away_team_stats['last10_goal_diff'],
        'rel_winrate': home_team_stats['last10_winrate'] - away_team_stats['last10_winrate'],
        'rel_momentum': home_team_stats['last3_winrate'] - away_team_stats['last3_winrate'],
        'rel_xg': home_team_stats['last10_xg'] - away_team_stats['last10_xg'],
        'rel_shots': home_team_stats['last10_shots'] - away_team_stats['last10_shots'],
        'home_advantage': 1.0
    }
    
    # Create DataFrame
    X_new = pd.DataFrame([input_data])[features]
    
    # Scale
    X_new_scaled = scaler.transform(X_new)
    
    # Predict
    proba = model.predict_proba(X_new_scaled)[0, 1]
    prediction = 'Home Win' if proba > 0.5 else 'Away Win'
    confidence = max(proba, 1 - proba)
    
    return {
        'prediction': prediction,
        'home_win_probability': proba,
        'confidence': confidence,
        'bet_recommendation': 'BET' if confidence >= 0.65 else 'NO BET'
    }

# Example usage:
print("\n" + "="*60)
print("EXAMPLE PREDICTION")
print("="*60)

example_home = {
    'elo': 1550,
    'last10_goal_diff': 0.5,
    'last10_winrate': 0.6,
    'last3_winrate': 0.67,
    'last10_xg': 3.2,
    'last10_shots': 28.5
}

example_away = {
    'elo': 1480,
    'last10_goal_diff': -0.3,
    'last10_winrate': 0.45,
    'last3_winrate': 0.33,
    'last10_xg': 2.8,
    'last10_shots': 26.0
}

result = predict_new_game(example_home, example_away)

print(f"\nPrediction: {result['prediction']}")
print(f"Home Win Probability: {result['home_win_probability']:.2%}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Recommendation: {result['bet_recommendation']}")

# ==========================
# 16. SUMMARY
# ==========================
end_time = time.time()

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Training time:    {end_time - start_time:.2f} seconds")
print(f"Model type:       Logistic Regression")
print(f"Features:         {len(features)}")
print(f"Test accuracy:    {test_acc:.2%}")
print(f"Test AUC:         {test_auc:.4f}")

if test_acc >= 0.60:
    print("\n✓✓ GOOD MODEL (≥60% accuracy)")
elif test_acc >= 0.55:
    print("\n✓ ACCEPTABLE MODEL (55-60% accuracy)")
else:
    print("\n⚠️  WEAK MODEL (<55% accuracy)")

print("\nNext steps:")
print("1. Use confidence filtering (≥0.65 threshold)")
print("2. Only bet on high-confidence predictions")
print("3. Track actual results over time")
print("4. Consider fixing ELO system if accuracy is low")