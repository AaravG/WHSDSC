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
# ELO RATING SYSTEM
# ==========================================
K = 30
INITIAL_RATING = 1500

teams = set(game_level["home_team"]).union(set(game_level["away_team"]))
elo = {team: INITIAL_RATING for team in teams}

home_elos = []
away_elos = []

for _, row in game_level.iterrows():
    home = row["home_team"]
    away = row["away_team"]

    Rh = elo[home]
    Ra = elo[away]

    home_elos.append(Rh)
    away_elos.append(Ra)

    Eh = 1 / (1 + 10 ** ((Ra - Rh) / 400))
    S = row["home_win"]

    elo[home] = Rh + K * (S - Eh)
    elo[away] = Ra + K * ((1 - S) - (1 - Eh))

game_level["home_elo"] = home_elos
game_level["away_elo"] = away_elos
game_level["elo_diff"] = game_level["home_elo"] - game_level["away_elo"]

# Feature engineering WITHOUT using goals (use xG as proxy)
game_level["xg_diff"] = game_level["home_xg"] - game_level["away_xg"]
game_level["shot_diff"] = game_level["home_shots"] - game_level["away_shots"]
game_level["assist_diff"] = game_level["home_assists"] - game_level["away_assists"]
game_level["penalty_diff"] = game_level["away_penalties_committed"] - game_level["home_penalties_committed"]
game_level["max_xg_diff"] = game_level["home_max_xg"] - game_level["away_max_xg"]

game_level["home_xg_per_shot"] = game_level["home_xg"] / game_level["home_shots"]
game_level["away_xg_per_shot"] = game_level["away_xg"] / game_level["away_shots"]
game_level["xg_per_shot_diff"] = game_level["home_xg_per_shot"] - game_level["away_xg_per_shot"]

game_level = game_level.fillna(0)
game_level = game_level.replace([np.inf, -np.inf], 0)

# ==========================================
# Train Model WITHOUT Data Leakage
# ==========================================
print("=" * 60)
print("Training Model (No Data Leakage)")
print("=" * 60)

feature_cols = [
    "xg_diff",
    "shot_diff",
    "assist_diff",
    "xg_per_shot_diff",
    "penalty_diff",
    "max_xg_diff"
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
else:
    print("\n✓ Accuracy looks realistic for predictive model")

print("\n" + classification_report(y_test, y_pred, target_names=["Away Win", "Home Win"]))

cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ==========================================
# Build Team Rankings
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

    wins = home["home_win"].sum() + (1 - away["home_win"]).sum()
    row["wins"] = int(wins)
    row["win_pct"] = wins / games_played

    row["goals_for"] = pd.concat([home["home_goals"], away["away_goals"]]).mean()
    row["goals_against"] = pd.concat([home["away_goals"], away["home_goals"]]).mean()
    row["goal_diff"] = row["goals_for"] - row["goals_against"]

    row["xg_for"] = pd.concat([home["home_xg"], away["away_xg"]]).mean()
    row["xg_against"] = pd.concat([home["away_xg"], away["home_xg"]]).mean()
    row["xg_diff"] = row["xg_for"] - row["xg_against"]

    row["shots_for"] = pd.concat([home["home_shots"], away["away_shots"]]).mean()
    row["shots_against"] = pd.concat([home["away_shots"], away["home_shots"]]).mean()
    row["shot_diff"] = row["shots_for"] - row["shots_against"]

    row["assists_for"] = pd.concat([home["home_assists"], away["away_assists"]]).mean()
    row["assists_against"] = pd.concat([home["away_assists"], away["home_assists"]]).mean()
    row["assist_diff"] = row["assists_for"] - row["assists_against"]

    row["penalty_for"] = pd.concat([home["home_penalties_committed"], away["away_penalties_committed"]]).mean()
    row["penalty_against"] = pd.concat([home["away_penalties_committed"], away["home_penalties_committed"]]).mean()
    row["penalty_diff"] = row["penalty_against"] - row["penalty_for"]

    row["max_xg_for"] = pd.concat([home["home_max_xg"], away["away_max_xg"]]).max()
    row["max_xg_against"] = pd.concat([home["away_max_xg"], away["home_max_xg"]]).max()
    row["max_xg_diff"] = row["max_xg_for"] - row["max_xg_against"]

    row["xg_per_shot"] = row["xg_for"] / row["shots_for"] if row["shots_for"] > 0 else 0
    row["opp_xg_per_shot"] = row["xg_against"] / row["shots_against"] if row["shots_against"] > 0 else 0
    row["xg_per_shot_diff"] = row["xg_per_shot"] - row["opp_xg_per_shot"]

    team_rows.append(row)

team_df = pd.DataFrame(team_rows)

# ==========================================
# Build Final Elo Table
# ==========================================
elo_df = pd.DataFrame([
    {"team": team, "elo_rating": rating}
    for team, rating in elo.items()
]).sort_values("elo_rating", ascending=False)

elo_df["rank"] = range(1, len(elo_df) + 1)

# ==========================================
# Apply Learned Weights to Rankings
# ==========================================
ranking_features = [
    "xg_diff",
    "shot_diff",
    "assist_diff",
    "xg_per_shot_diff",
    "penalty_diff",
    "max_xg_diff"
]

scaler = StandardScaler()
team_df_scaled = team_df.copy()
team_df_scaled[ranking_features] = scaler.fit_transform(team_df[ranking_features])

team_df["composite_score"] = sum(
    team_df_scaled[feat] * learned_weights[feat]
    for feat in ranking_features
)

# ==========================================
# Add Final Elo
# ==========================================
team_df = team_df.merge(
    elo_df[["team", "elo_rating"]],
    on="team",
    how="left"
)

team_df["elo_scaled"] = StandardScaler().fit_transform(team_df[["elo_rating"]])

# ==========================================
# Find Best Alpha (NOW composite_score and elo_scaled both exist)
# ==========================================
alphas = np.linspace(0, 1, 21)
results = []

for a in alphas:
    hybrid = a * team_df["elo_scaled"] + (1 - a) * team_df["composite_score"]
    corr = hybrid.corr(team_df["win_pct"])
    results.append((a, corr))

best_alpha, best_corr = max(results, key=lambda x: x[1])

print(f"Best Alpha (Elo weight): {best_alpha:.2f}")
print(f"Correlation with Win %: {best_corr:.4f}")

# ==========================================
# HYBRID SCORE
# ==========================================
team_df["hybrid_score"] = (
    best_alpha * team_df["elo_scaled"] +
    (1 - best_alpha) * team_df["composite_score"]
)

rankings_hybrid = team_df.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
rankings_hybrid["rank"] = range(1, len(rankings_hybrid) + 1)

print("\nHYBRID RANKINGS")
print(rankings_hybrid[["rank", "team", "hybrid_score", "elo_rating", "composite_score"]])

# ==========================================
# PURE ELO RANKINGS
# ==========================================
print("\nPURE ELO RANKINGS")
print(elo_df)

# ==========================================
# PURE MODEL RANKINGS
# ==========================================
rankings_model = team_df.sort_values("composite_score", ascending=False).reset_index(drop=True)
rankings_model["rank"] = range(1, len(rankings_model) + 1)

print("\nMODEL-WEIGHTED RANKINGS")
print(rankings_model[["rank", "team", "composite_score", "win_pct", "goal_diff"]])

# ==========================================
# Save Results
# ==========================================
rankings_model.to_csv('team_rankings_model.csv', index=False)

weights_df = pd.DataFrame([
    {"feature": feat, "weight": weight, "importance_pct": weight * 100}
    for feat, weight in learned_weights.items()
]).sort_values("weight", ascending=False)
weights_df.to_csv('learned_feature_weights.csv', index=False)

plt.figure(figsize=(10, 6))
features_sorted = sorted(learned_weights.items(), key=lambda x: x[1], reverse=True)
features, weights = zip(*features_sorted)
plt.barh(range(len(features)), [w * 100 for w in weights], color='steelblue')
plt.yticks(range(len(features)), features)
plt.xlabel('Importance (%)')
plt.title('Feature Importance (No Data Leakage)')
plt.tight_layout()
plt.savefig('feature_importance_fixed.png', dpi=300, bbox_inches='tight')

print("\n" + "=" * 60)
print("RESULTS SAVED")
print("=" * 60)
print("✓ team_rankings_model.csv")
print("✓ learned_feature_weights.csv")
print("✓ feature_importance_fixed.png")
print("✓ team_rankings_elo (DB)")
print("✓ team_rankings_hybrid (DB)")

time.sleep(3)

conn = sqlite3.connect(
    r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db",
    timeout=30
)
conn.execute("PRAGMA journal_mode=WAL;")

rankings_model.to_sql("team_rankings_model", conn, if_exists="replace", index=False)
weights_df.to_sql("learned_feature_weights", conn, if_exists="replace", index=False)
elo_df.to_sql("team_rankings_elo", conn, if_exists="replace", index=False)
rankings_hybrid.to_sql("team_rankings_hybrid", conn, if_exists="replace", index=False)

conn.commit()
conn.close()

numeric_df = game_level.select_dtypes(include=[np.number])
print(numeric_df.corr()["home_win"].sort_values(ascending=False))
print(X.var())
print(X.corr())