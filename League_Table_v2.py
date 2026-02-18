import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import sqlite3
import time



# ==========================================
# Load and Aggregate Data
# ==========================================
conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")
df = pd.read_sql("SELECT * FROM kNN", conn)
conn.close()




game_level = df.groupby('game_id').agg({
    'home_team': 'first',
    'away_team': 'first',
    'went_ot': 'first',
    'home_win': 'first',
    'home_assists': 'sum',
    'home_shots': 'sum',
    'home_goals': 'sum',
    'away_assists': 'sum',
    'away_shots': 'sum',
    'away_goals': 'sum',
    'home_penalties_committed': 'sum',
    'home_penalty_minutes': 'sum',
    'away_penalties_committed': 'sum',
    'away_penalty_minutes': 'sum',
    'home_xg': 'sum',
    'away_xg': 'sum',
    'home_max_xg': 'max',
    'away_max_xg': 'max',
    'toi': 'sum'
}).reset_index()

# ==========================================
# CRITICAL FIX: Remove Data Leakage
# ==========================================
# DON'T use goal_diff for prediction - it contains the answer!
# Instead, use these features that represent team quality:

# Feature engineering WITHOUT using goals (use xG as proxy)
game_level["xg_diff"] = game_level["home_xg"] - game_level["away_xg"]
game_level["shot_diff"] = game_level["home_shots"] - game_level["away_shots"]
game_level["assist_diff"] = game_level["home_assists"] - game_level["away_assists"]
game_level["penalty_diff"] = game_level["away_penalty_minutes"] - game_level["home_penalty_minutes"]

# Shooting efficiency (based on xG, not actual goals)
game_level["home_xg_per_shot"] = game_level["home_xg"] / game_level["home_shots"]
game_level["away_xg_per_shot"] = game_level["away_xg"] / game_level["away_shots"]
game_level["xg_per_shot_diff"] = game_level["home_xg_per_shot"] - game_level["away_xg_per_shot"]

# Handle divisions by zero
game_level = game_level.fillna(0)
game_level = game_level.replace([np.inf, -np.inf], 0)

# ==========================================
# Train Model WITHOUT Data Leakage
# ==========================================
print("=" * 60)
print("Training Model (No Data Leakage)")
print("=" * 60)

feature_cols = [
    "xg_diff",           # Expected goal quality
    "shot_diff",         # Shot volume
    "assist_diff",       # Playmaking
    "penalty_diff",      # Discipline
    "xg_per_shot_diff"   # Shot quality
]

X = game_level[feature_cols]
y = game_level["home_win"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Extract weights
perm = permutation_importance(
    rf_model, X_test, y_test,
    n_repeats=20,
    random_state=42,
    n_jobs=-1
)

importance_scores = perm.importances_mean

total_importance = sum(importance_scores)
learned_weights = {
    feat: score / total_importance 
    for feat, score in zip(feature_cols, importance_scores)
}

print("\n" + "=" * 60)
print("LEARNED FEATURE WEIGHTS")
print("=" * 60)
for feat, weight in sorted(learned_weights.items(), key=lambda x: x[1], reverse=True):
    print(f"{feat:30s}: {weight:.4f} ({weight*100:.1f}%)")

# Performance
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

test_accuracy = rf_model.score(X_test, y_test)
train_accuracy = rf_model.score(X_train, y_train)
auc_score = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE (Should be 70-85%, NOT 100%)")
print("=" * 60)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy:     {test_accuracy:.4f}")
print(f"AUC-ROC Score:     {auc_score:.4f}")

if test_accuracy > 0.95:
    print("\n⚠️  WARNING: Accuracy > 95% suggests data leakage!")
    print("   Model should be 70-85% for legitimate prediction")
else:
    print("\n✓ Accuracy looks realistic for predictive model")

print("\n" + classification_report(y_test, y_pred, target_names=["Away Win", "Home Win"]))

cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ==========================================
# Build Team Rankings (This part is fine)
# ==========================================
print("\n" + "=" * 60)
print("Computing Team Rankings")
print("=" * 60)

teams = sorted(set(game_level["home_team"]).union(set(game_level["away_team"])))
team_rows = []

for team in teams:
    home = game_level[game_level["home_team"] == team]
    away = game_level[game_level["away_team"] == team]
    games_played = len(home) + len(away)
    
    if games_played == 0:
        continue
    
    row = {"team": team, "games_played": games_played}
    
    # Win percentage
    wins = home["home_win"].sum() + (1 - away["home_win"]).sum()
    row["wins"] = int(wins)
    row["win_pct"] = wins / games_played
    
    # NOW we can use actual goals for ranking (not prediction)
    row["goals_for"] = pd.concat([home["home_goals"], away["away_goals"]]).mean()
    row["goals_against"] = pd.concat([home["away_goals"], away["home_goals"]]).mean()
    row["goal_diff"] = row["goals_for"] - row["goals_against"]  # OK for rankings!
    
    # Other stats
    row["xg_for"] = pd.concat([home["home_xg"], away["away_xg"]]).mean()
    row["xg_against"] = pd.concat([home["away_xg"], away["home_xg"]]).mean()
    row["xg_diff"] = row["xg_for"] - row["xg_against"]
    
    row["shots_for"] = pd.concat([home["home_shots"], away["away_shots"]]).mean()
    row["shots_against"] = pd.concat([home["away_shots"], away["home_shots"]]).mean()
    row["shot_diff"] = row["shots_for"] - row["shots_against"]
    
    row["assists_for"] = pd.concat([home["home_assists"], away["away_assists"]]).mean()
    row["assists_against"] = pd.concat([home["away_assists"], away["home_assists"]]).mean()
    row["assist_diff"] = row["assists_for"] - row["assists_against"]
    
    row["penalty_minutes_for"] = pd.concat([home["home_penalty_minutes"], away["away_penalty_minutes"]]).mean()
    row["penalty_minutes_against"] = pd.concat([home["away_penalty_minutes"], away["home_penalty_minutes"]]).mean()
    row["penalty_min_diff"] = row["penalty_minutes_against"] - row["penalty_minutes_for"]
    
    # Shot quality
    row["xg_per_shot"] = row["xg_for"] / row["shots_for"] if row["shots_for"] > 0 else 0
    row["opp_xg_per_shot"] = row["xg_against"] / row["shots_against"] if row["shots_against"] > 0 else 0
    row["xg_per_shot_diff"] = row["xg_per_shot"] - row["opp_xg_per_shot"]
    
    
    
    team_rows.append(row)

team_df = pd.DataFrame(team_rows)

# ==========================================
# Apply Learned Weights to Rankings
# ==========================================
ranking_features = ["xg_diff", "shot_diff", "assist_diff", "penalty_min_diff", "xg_per_shot_diff"]


scaler = StandardScaler()
team_df_scaled = team_df.copy()
team_df_scaled[ranking_features] = scaler.fit_transform(team_df[ranking_features])

# Composite score from model weights
team_df["composite_score"] = sum(
    team_df_scaled[feat] * learned_weights[feat]
    for feat in ranking_features
)

# ==========================================
# Manual Weighted Rankings (Alternative)
# # ==========================================
# # If model weights don't make sense, use manual weights based on hockey analytics:
# manual_weights = {
#     "goal_diff": 0.40,      # Most important
#     "xg_diff": 0.25,
#     "shot_diff": 0.15,
#     "assist_diff": 0.10,
#     "penalty_diff": 0.05,
#     "xg_per_shot_diff": 0.05
# }

# # For manual rankings, we CAN use goal_diff (it's historical data)
# team_df_manual = team_df.copy()
# manual_features = ["goal_diff", "xg_diff", "shot_diff", "assist_diff", "penalty_diff", "xg_per_shot_diff"]
# team_df_manual[manual_features] = scaler.fit_transform(team_df[manual_features])

# team_df["manual_score"] = sum(
#     team_df_manual[feat] * manual_weights[feat]
#     for feat in manual_features
# )

# ==========================================
# Final Rankings (Two Methods)
# ==========================================
rankings_model = team_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
rankings_model["rank"] = range(1, len(rankings_model) + 1)

# rankings_manual = team_df.sort_values("manual_score", ascending=False).reset_index(drop=True)
# rankings_manual["rank"] = range(1, len(rankings_manual) + 1)

print("\n" + "=" * 60)
print("METHOD 1: Model-Weighted Rankings (from predictive features)")
print("=" * 60)
print(rankings_model[[
    "rank", "team", "composite_score", "wins", "games_played", 
    "win_pct", "goal_diff", "xg_diff"
]].to_string(index=False))

correlation_model = rankings_model["composite_score"].corr(rankings_model["win_pct"])
print(f"\nCorrelation with Win %: {correlation_model:.4f}")

# print("\n" + "=" * 60)
# print("METHOD 2: Manual-Weighted Rankings (includes actual goals)")
# print("=" * 60)
# print(rankings_manual[[
#     "rank", "team", "manual_score", "wins", "games_played", 
#     "win_pct", "goal_diff", "xg_diff"
# ]].head(15).to_string(index=False))

# correlation_manual = rankings_manual["manual_score"].corr(rankings_manual["win_pct"])
# print(f"\nCorrelation with Win %: {correlation_manual:.4f}")

# ==========================================
# Save Results
# ==========================================
rankings_model.to_csv('team_rankings_model.csv', index=False)
# rankings_manual.to_csv('team_rankings_manual.csv', index=False)

weights_df = pd.DataFrame([
    {"feature": feat, "weight": weight, "importance_pct": weight * 100}
    for feat, weight in learned_weights.items()
]).sort_values("weight", ascending=False)
weights_df.to_csv('learned_feature_weights.csv', index=False)

# Visualization
plt.figure(figsize=(10, 6))
features_sorted = sorted(learned_weights.items(), key=lambda x: x[1], reverse=True)
features, weights = zip(*features_sorted)

plt.barh(range(len(features)), [w*100 for w in weights], color='steelblue')
plt.yticks(range(len(features)), features)
plt.xlabel('Importance (%)')
plt.title('Feature Importance (No Data Leakage)')
plt.tight_layout()
plt.savefig('feature_importance_fixed.png', dpi=300, bbox_inches='tight')

print("\n" + "=" * 60)
print("RESULTS SAVED")
print("=" * 60)
print("✓ team_rankings_model.csv  → Rankings from ML weights")
print("✓ team_rankings_manual.csv → Rankings from manual weights")
print("✓ learned_feature_weights.csv")
print("✓ feature_importance_fixed.png")

time.sleep(1)  # give OS time to release any file handles

conn = sqlite3.connect(
    r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db",
    timeout=30  # wait up to 30s if locked
)

conn.execute("PRAGMA journal_mode=WAL;")  # better concurrency

rankings_model.to_sql(
    "team_rankings_model",
    conn,
    if_exists="replace",
    index=False
)

weights_df.to_sql(
    "learned_feature_weights",
    conn,
    if_exists="replace",
    index=False
)

conn.commit()
conn.close()