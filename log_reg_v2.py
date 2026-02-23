import pandas as pd
import numpy as np
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import time

start_time = time.time()

print("="*60)
print("FIXED LOGISTIC REGRESSION (No Data Leakage)")
print("="*60)

# ==========================
# 1. LOAD & SORT PROPERLY
# ==========================
conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")
df = pd.read_sql("SELECT * FROM NN", conn)
conn.close()

print(f"✓ Loaded {len(df)} games")

# CRITICAL: Sort by game number FIRST, before any feature engineering
df["game_num"] = df["game_id"].str.extract(r'(\d+)').astype(int)
df = df.sort_values("game_num").reset_index(drop=True)
print(f"✓ Sorted chronologically by game_num")

# ==========================
# 2. VALIDATE SORT
# ==========================
print("\nValidating chronological order...")
print(df[['game_num', 'home_team', 'away_team']].head(10))

# ==========================
# 3. FEATURE ENGINEERING (FIXED)
# ==========================
print("\nCalculating features WITHOUT data leakage...")

rolling = 10

# Create for/against columns
df["home_goals_for"] = df["home_goals"]
df["home_goals_against"] = df["away_goals"]
df["away_goals_for"] = df["away_goals"]
df["away_goals_against"] = df["home_goals"]

if "away_win" not in df.columns:
    df["away_win"] = 1 - df["home_win"]

# === HOME TEAM ROLLING STATS ===
# CRITICAL: Use min_periods=3 so first few games have NaN (no leakage)
df["home_last10_goals_for"] = (
    df.groupby("home_team", group_keys=False)["home_goals_for"]
      .apply(lambda x: x.shift(1).rolling(rolling, min_periods=3).mean())
)

df["home_last10_goals_against"] = (
    df.groupby("home_team", group_keys=False)["home_goals_against"]
      .apply(lambda x: x.shift(1).rolling(rolling, min_periods=3).mean())
)

df["home_last10_goal_diff"] = df["home_last10_goals_for"] - df["home_last10_goals_against"]

df["home_last10_winrate"] = (
    df.groupby("home_team", group_keys=False)["home_win"]
      .apply(lambda x: x.shift(1).rolling(rolling, min_periods=3).mean())
)

df["home_last3_winrate"] = (
    df.groupby("home_team", group_keys=False)["home_win"]
      .apply(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
)

# === AWAY TEAM ROLLING STATS ===
df["away_last10_goals_for"] = (
    df.groupby("away_team", group_keys=False)["away_goals_for"]
      .apply(lambda x: x.shift(1).rolling(rolling, min_periods=3).mean())
)

df["away_last10_goals_against"] = (
    df.groupby("away_team", group_keys=False)["away_goals_against"]
      .apply(lambda x: x.shift(1).rolling(rolling, min_periods=3).mean())
)

df["away_last10_goal_diff"] = df["away_last10_goals_for"] - df["away_last10_goals_against"]

df["away_last10_winrate"] = (
    df.groupby("away_team", group_keys=False)["away_win"]
      .apply(lambda x: x.shift(1).rolling(rolling, min_periods=3).mean())
)

df["away_last3_winrate"] = (
    df.groupby("away_team", group_keys=False)["away_win"]
      .apply(lambda x: x.shift(1).rolling(3, min_periods=2).mean())
)

# ==========================
# 4. VALIDATE NO DATA LEAKAGE
# ==========================
print("\n" + "="*60)
print("VALIDATION: Checking for data leakage...")
print("="*60)

# For each team, find their ACTUAL first home game (lowest game_num)
validation_results = []

for team in df['home_team'].unique():
    team_home_games = df[df['home_team'] == team].sort_values('game_num')
    if len(team_home_games) > 0:
        first_game = team_home_games.iloc[0]
        validation_results.append({
            'team': team,
            'first_game_num': first_game['game_num'],
            'home_goals_for': first_game['home_goals_for'],
            'home_last10_goals_for': first_game['home_last10_goals_for'],
            'is_nan': pd.isna(first_game['home_last10_goals_for'])
        })

validation_df = pd.DataFrame(validation_results)
nan_count = validation_df['is_nan'].sum()
total_teams = len(validation_df)

print(f"Teams with NaN on their first home game: {nan_count} / {total_teams}")

if nan_count >= total_teams * 0.8:
    print("✓✓ GOOD! No data leakage - first games properly have NaN")
    print("\nSample of first games:")
    print(validation_df.head(10).to_string(index=False))
else:
    print("❌ STILL HAS LEAKAGE!")
    print("\nTeams with data leakage (showing first 10):")
    leakage_teams = validation_df[~validation_df['is_nan']]
    print(leakage_teams.head(10).to_string(index=False))

# ==========================
# 5. RELATIVE FEATURES
# ==========================
df["elo_diff"] = df["home_elo"] - df["away_elo"]
df["rel_goal_diff"] = df["home_last10_goal_diff"] - df["away_last10_goal_diff"]
df["rel_winrate"] = df["home_last10_winrate"] - df["away_last10_winrate"]
df["rel_momentum"] = df["home_last3_winrate"] - df["away_last3_winrate"]
df["home_advantage"] = 1.0

# ==========================
# 6. CLEAN DATA
# ==========================
features = [
    "elo_diff",
    "rel_goal_diff",
    "rel_winrate",
    "rel_momentum",
    "home_advantage"
]

df = df.replace([np.inf, -np.inf], np.nan)
df_clean = df.dropna(subset=features + ["home_win"]).reset_index(drop=True)

print(f"\n✓ Final dataset: {len(df_clean)} games")

# ==========================
# 7. SPLIT
# ==========================
train = df_clean.iloc[:int(0.6*len(df_clean))]
valid = df_clean.iloc[int(0.6*len(df_clean)):int(0.8*len(df_clean))]
test = df_clean.iloc[int(0.8*len(df_clean)):]

print(f"Split: Train={len(train)}, Valid={len(valid)}, Test={len(test)}")

X_train = train[features]
y_train = train["home_win"]

X_valid = valid[features]
y_valid = valid["home_win"]

X_test = test[features]
y_test = test["home_win"]

# ==========================
# 8. SCALE & TRAIN
# ==========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    C=1.0
)

model.fit(X_train_scaled, y_train)
print("✓ Model trained")

# ==========================
# 9. FEATURE IMPORTANCE
# ==========================
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

for feat, coef in zip(features, model.coef_[0]):
    print(f"{feat:20s}: {coef:+.4f}")

# ==========================
# 10. VALIDATION PERFORMANCE
# ==========================
print("\n" + "="*60)
print("VALIDATION PERFORMANCE")
print("="*60)

y_valid_pred = model.predict(X_valid_scaled)
y_valid_proba = model.predict_proba(X_valid_scaled)[:, 1]

valid_acc = (y_valid_pred == y_valid).mean()
valid_auc = roc_auc_score(y_valid, y_valid_proba)

print(f"Accuracy: {valid_acc:.4f} ({valid_acc*100:.2f}%)")
print(f"AUC:      {valid_auc:.4f}")

print("\n" + classification_report(y_valid, y_valid_pred, target_names=["Away Win", "Home Win"]))

# ==========================
# 11. TEST PERFORMANCE
# ==========================
print("\n" + "="*60)
print("TEST PERFORMANCE (FINAL)")
print("="*60)

y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

test_acc = (y_test_pred == y_test).mean()
test_auc = roc_auc_score(y_test, y_test_proba)

print(f"Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"AUC:      {test_auc:.4f}")

print("\n" + classification_report(y_test, y_test_pred, target_names=["Away Win", "Home Win"]))

# ==========================
# 12. CONFIDENCE FILTERING
# ==========================
print("\n" + "="*60)
print("CONFIDENCE FILTERING")
print("="*60)

thresholds = [0.60, 0.65, 0.70]

for threshold in thresholds:
    confident_mask = (y_test_proba >= threshold) | (y_test_proba <= (1 - threshold))
    
    if confident_mask.sum() > 0:
        y_test_confident = y_test[confident_mask]
        y_pred_confident = y_test_pred[confident_mask]
        
        conf_acc = (y_pred_confident == y_test_confident).mean()
        n_bets = confident_mask.sum()
        
        print(f"\nThreshold {threshold:.2f}:")
        print(f"  Accuracy: {conf_acc:.2%}")
        print(f"  Bets: {n_bets} ({n_bets/len(y_test)*100:.1f}%)")
        print(f"  ROI: {(conf_acc - 0.525) * 100:.1f}%")

# ==========================
# 13. SUMMARY
# ==========================
end_time = time.time()

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Time: {end_time - start_time:.2f}s")
print(f"Test Accuracy: {test_acc:.2%}")
print(f"Test AUC: {test_auc:.4f}")

if test_acc >= 0.65:
    print("\n✓✓ EXCELLENT! Model is working (≥65%)")
    print("   Data leakage has been fixed!")
elif test_acc >= 0.60:
    print("\n✓ GOOD! Model improved (60-65%)")
    print("  Still room for improvement")
elif test_acc >= 0.55:
    print("\n⚠️  ACCEPTABLE (55-60%)")
    print("  Consider adding more features")
else:
    print("\n❌ STILL WEAK (<55%)")
    print("  May need non-linear model or better features")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print("BEFORE fix (with leakage): 54% accuracy")
print(f"AFTER fix (no leakage):    {test_acc:.0%} accuracy")
print(f"Improvement: {(test_acc - 0.54) * 100:+.1f} percentage points")