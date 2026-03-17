import pandas as pd
import numpy as np
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import warnings 

warnings.filterwarnings("ignore", category=UserWarning)

# 1. DATABASE CONNECTION
db_path = r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db"
conn = sqlite3.connect(db_path)

df_history = pd.read_sql("SELECT home_team, away_team, home_win FROM NN", conn)
df_comp = pd.read_sql("SELECT home_team, away_team FROM Matchups", conn)

# Change this line:
df_rankings = pd.read_sql("""
    SELECT 
        home_team as team, 
        hybrid_score, 
        current_elo as elo_rating, 
        efficiency_score 
    FROM final_rankings_hybrid_v5
""", conn)

# Update the lookup maps to match the new column names:
h_map = dict(zip(df_rankings['team'], df_rankings['hybrid_score']))
e_map = dict(zip(df_rankings['team'], df_rankings['elo_rating']))
# Note: Since 'composite_score' was from an old version, you can either 
# remove it or use 'efficiency_score' as its replacement:
c_map = dict(zip(df_rankings['team'], df_rankings['efficiency_score']))

# 3. DATA PREPARATION
features = []
for _, row in df_history.iterrows():
    h, a = row['home_team'], row['away_team']
    if h in h_map and a in h_map:
        features.append([
            h_map[h] - h_map[a],
            e_map[h] - e_map[a],
            c_map[h] - c_map[a],
            row['home_win']
        ])

df_all = pd.DataFrame(features, columns=['h_diff', 'e_diff', 'c_diff', 'target'])
X = df_all[['h_diff', 'e_diff', 'c_diff']]
y = df_all['target']

# 4. STRATIFIED K-FOLD OOF STACKING
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros((len(df_all), 3))

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train = y.iloc[train_idx]

    m_h = LogisticRegression().fit(X_train[['h_diff']], y_train)
    m_e = LogisticRegression().fit(X_train[['e_diff']], y_train)
    m_c = LogisticRegression().fit(X_train[['c_diff']], y_train)

    oof_preds[val_idx, 0] = m_h.predict_proba(X_val[['h_diff']])[:, 1]
    oof_preds[val_idx, 1] = m_e.predict_proba(X_val[['e_diff']])[:, 1]
    oof_preds[val_idx, 2] = m_c.predict_proba(X_val[['c_diff']])[:, 1]

# 5. META-MODEL (Stacked Calibrator)
meta_model = LogisticRegression().fit(oof_preds, y)

# Retrain base models on full data
final_m_h = LogisticRegression().fit(X[['h_diff']], y)
final_m_e = LogisticRegression().fit(X[['e_diff']], y)
final_m_c = LogisticRegression().fit(X[['c_diff']], y)

# 6. GENERATE COMPETITION RESULTS
results = []

for _, row in df_comp.iterrows():
    h, a = row['home_team'], row['away_team']

    if h not in h_map or a not in h_map:
        continue

    h_diff = h_map[h] - h_map[a]
    e_diff = e_map[h] - e_map[a]
    c_diff = c_map[h] - c_map[a]

    p_h = final_m_h.predict_proba([[h_diff]])[0][1]
    p_e = final_m_e.predict_proba([[e_diff]])[0][1]
    p_c = final_m_c.predict_proba([[c_diff]])[0][1]

    final_prob = meta_model.predict_proba([[p_h, p_e, p_c]])[0][1]
    recommendation = h if final_prob >= 0.5 else a

    results.append((
        h_map[h], h_map[a], p_h,
        e_map[h], e_map[a], p_e,
        c_map[h], c_map[a], p_c,
        final_prob, recommendation,
        h, a
    ))

# 7. DB WRITEBACK
cursor = conn.cursor()

update_sql = """
UPDATE Matchups SET
home_hybrid_score = ?, away_hybrid_score = ?, hybrid_win_prob = ?,
home_elo_score = ?, away_elo_score = ?, elo_win_prob = ?,
home_composite_score = ?, away_composite_score = ?, composite_win_prob = ?,
final_win_prob = ?, final_recommendation = ?
WHERE home_team = ? AND away_team = ?
"""

cursor.executemany(update_sql, results)
conn.commit()


print("Phase 1A updated with Stratified Stacked Ensemble.")

# ==========================
# 8. ACCURACY JUDGEMENT (BACKTESTING)
# ==========================
from sklearn.metrics import accuracy_score, log_loss

# 1. Individual Base Model Accuracies (using OOF predictions for fairness)
acc_h = accuracy_score(y, (oof_preds[:, 0] >= 0.5).astype(int))
acc_e = accuracy_score(y, (oof_preds[:, 1] >= 0.5).astype(int))
acc_c = accuracy_score(y, (oof_preds[:, 2] >= 0.5).astype(int))

# 2. Meta-Model (Stacked) Accuracy
# We use the OOF preds to see how the meta-model would have performed on unseen data
meta_oof_probs = meta_model.predict_proba(oof_preds)[:, 1]
acc_meta = accuracy_score(y, (meta_oof_probs >= 0.5).astype(int))

# 3. Log-Loss (Lower is better - measures probability calibration)
loss_meta = log_loss(y, meta_oof_probs)

print("\n" + "="*30)
print(" MODEL PERFORMANCE SUMMARY")
print("="*30)
print(f"Hybrid Base Accuracy:     {acc_h:.2%}")
print(f"Elo Base Accuracy:        {acc_e:.2%}")
print(f"Efficiency Base Accuracy: {acc_c:.2%}")
print("-" * 30)
print(f"STACKED META-ACCURACY:    {acc_meta:.2%}")
print(f"Stacked Log-Loss:         {loss_meta:.4f}")
print("="*30)

if acc_meta > max(acc_h, acc_e, acc_c):
    print("STATUS: Meta-Model successfully improved predictive power.")
else:
    print("STATUS: Meta-Model performing at par with best base model.")