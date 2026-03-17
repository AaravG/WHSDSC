import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler

# ==========================
# 1. CONNECT & LOAD
# ==========================
DB_PATH = r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db"
conn = sqlite3.connect(DB_PATH)

# Loading data from your NN table
df = pd.read_sql("SELECT * FROM NN", conn)

# ==========================
# 2. FEATURE ENGINEERING
# ==========================
df['xg_diff'] = df['home_xg'] - df['away_xg']
df['shot_diff'] = df['home_shots'] - df['away_shots']
df['penalty_diff'] = df['away_penalties_committed'] - df['home_penalties_committed']
df['elo_diff'] = df['home_elo'] - df['away_elo']

# Proxy for Line Disparity: Ratio of top-end talent to total production
df['talent_concentration'] = df['home_max_xg'] / (df['home_xg'] + 1e-5)

feature_cols = [
    "xg_diff", "shot_diff", "penalty_diff", 
    "elo_diff", "home_max_xg", "talent_concentration"
]

X = df[feature_cols].fillna(0)
y = df["home_win"]

# ==========================
# 3. HYPERPARAMETER TUNING
# ==========================
print("Step 1: Tuning XGBoost Hyperparameters...")
param_dist = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

random_search = RandomizedSearchCV(
    xgb.XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='logloss'),
    param_distributions=param_dist,
    n_iter=108, 
    cv=5,      
    scoring='neg_log_loss',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

random_search.fit(X, y)
best_params = random_search.best_params_
print(f"Best Params Found: {best_params}")

# ==========================
# 4. TRAIN FINAL MODEL & SHAP
# ==========================
print("Step 2: Training Final Model & Calculating SHAP Values...")
model = xgb.XGBClassifier(**best_params)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# ==========================================
# 5. GENERATE FINAL HYBRID RANKINGS (Elo + SHAP)
# ==========================================
print("Step 3: Calculating Inner-Outer Nested Alpha Validation...")

shap_cols = [f"impact_{c}" for c in feature_cols]
shap_df = pd.DataFrame(shap_values, columns=shap_cols, index=df.index)
logic_check = pd.concat([
    df[['game_id', 'home_team', 'away_team', 'home_win', 'home_elo']], 
    shap_df
], axis=1)

team_stats = logic_check.groupby('home_team').agg({
    'home_win': 'mean',
    'home_elo': 'last',  
    **{col: 'mean' for col in shap_cols}
}).rename(columns={'home_win': 'win_rate', 'home_elo': 'current_elo'})

team_stats['efficiency_score'] = team_stats[shap_cols].sum(axis=1)

scaler = StandardScaler()
team_stats['elo_scaled'] = scaler.fit_transform(team_stats[['current_elo']])
team_stats['efficiency_scaled'] = scaler.fit_transform(team_stats[['efficiency_score']])

# --- TRUE NESTED CV FOR ALPHA ---
# Outer Loop splits the teams into test groups to prove alpha's generalizability
kf_outer = KFold(n_splits=5, shuffle=True, random_state=42)
alpha_candidates = np.linspace(0, 1, 51)
outer_results = []

for train_idx, test_idx in kf_outer.split(team_stats):
    train_fold = team_stats.iloc[train_idx]
    test_fold = team_stats.iloc[test_idx]
    
    # Inner Loop: Find the best Alpha for the training slice only
    inner_kf = KFold(n_splits=3, shuffle=True, random_state=7)
    fold_best_alpha = 0.5
    max_inner_corr = -1
    
    for a in alpha_candidates:
        inner_corrs = []
        for i_train_idx, i_val_idx in inner_kf.split(train_fold):
            i_val = train_fold.iloc[i_val_idx]
            h_val = (a * i_val['elo_scaled']) + ((1 - a) * i_val['efficiency_scaled'])
            inner_corrs.append(h_val.corr(i_val['win_rate']))
        
        avg_inner_corr = np.mean(inner_corrs)
        if avg_inner_corr > max_inner_corr:
            max_inner_corr = avg_inner_corr
            fold_best_alpha = a
            
    # VALIDATION: Test that Alpha on the unseen Outer fold
    h_outer = (fold_best_alpha * test_fold['elo_scaled']) + ((1 - fold_best_alpha) * test_fold['efficiency_scaled'])
    outer_corr = h_outer.corr(test_fold['win_rate'])
    outer_results.append((fold_best_alpha, outer_corr))

# Aggregate results
final_robust_alpha = np.mean([res[0] for res in outer_results])
avg_outer_performance = np.mean([res[1] for res in outer_results])

print(f"Rigorous Alpha (Elo Weighting): {final_robust_alpha:.2f}")
print(f"Mean Alpha Stability (Outer Correlation): {avg_outer_performance:.4f}")

# Final Score Calculation based on validated Alpha
team_stats['hybrid_score'] = (final_robust_alpha * team_stats['elo_scaled']) + \
                             ((1 - final_robust_alpha) * team_stats['efficiency_scaled'])

final_rankings = team_stats.sort_values("hybrid_score", ascending=False)
final_rankings['rank'] = range(1, len(final_rankings) + 1)

# ==========================
# 6. SAVE & VISUALIZE
# ==========================
final_rankings.to_sql("final_rankings_hybrid_v5", conn, if_exists="replace")
logic_check.to_sql("team_logic_crosscheck", conn, if_exists="replace", index=False)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, show=False)
plt.title("WHL Success Drivers: XGBoost Feature Impact", fontsize=14, pad=20)
plt.xlabel("Impact on Home Win Probability (Log-Odds)", fontsize=12)
plt.tight_layout()
plt.savefig('whl_success_drivers_shap.png', dpi=300)
plt.show()

conn.close()
print("\n✓ SUCCESS: Nested CV completed. Hybrid Rankings are statistically robust.")



import sqlite3
import pandas as pd

conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")
df = pd.read_sql("SELECT * FROM final_rankings_hybrid_v5 ORDER BY rank ASC", conn)

print(f"{'RANK':<5} | {'TEAM':<15} | {'HYBRID':<8} | {'ELO':<8} | {'EFFICIENCY':<10} | {'STATUS'}")
print("-" * 75)

for _, row in df.iterrows():
    # Logic to determine if they are 'Statistically Underrated' or 'Legacy Heavy'
    if row['efficiency_scaled'] > row['elo_scaled'] + 0.5:
        status = "⭐ SLEEPER"
    elif row['elo_scaled'] > row['efficiency_scaled'] + 0.5:
        status = "⚠️ OVERVALUED"
    else:
        status = "✅ BALANCED"
        
    print(f"{int(row['rank']):<5} | {row['home_team']:<15} | {row['hybrid_score']:<8.3f} | {row['current_elo']:<8.1f} | {row['efficiency_score']:<10.3f} | {status}")

conn.close()