import pandas as pd
import sqlite3
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. DATABASE CONNECTION
db_path = r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db"
conn = sqlite3.connect(db_path)

df_history = pd.read_sql("SELECT home_team, away_team, home_win FROM NN", conn)
df_comp = pd.read_sql("SELECT home_team, away_team FROM Matchups", conn)

# Change this line:
# Updated SQL to include efficiency_score
df_rankings = pd.read_sql("""
    SELECT 
        home_team as team, 
        hybrid_score, 
        current_elo as elo_rating, 
        efficiency_score 
    FROM final_rankings_hybrid_v5
""", conn)

# Update the lookup maps
hybrid_map = dict(zip(df_rankings['team'], df_rankings['hybrid_score']))
elo_map = dict(zip(df_rankings['team'], df_rankings['elo_rating']))
# Map efficiency_score to the 'c_map' (Composite Map) variable
comp_map = dict(zip(df_rankings['team'], df_rankings['efficiency_score']))

# 3. TRAIN & EVALUATE MODELS
history_data = []
for _, row in df_history.iterrows():
    h, a = row['home_team'], row['away_team']
    if all(t in hybrid_map for t in [h, a]):
        history_data.append({
            'hybrid_diff': hybrid_map[h] - hybrid_map[a],
            'elo_diff': elo_map[h] - elo_map[a],
            'comp_diff': comp_map[h] - comp_map[a],
            'actual_win': row['home_win']
        })

df_train = pd.DataFrame(history_data)

model_hyb = LogisticRegression().fit(df_train[['hybrid_diff']], df_train['actual_win'])
model_elo = LogisticRegression().fit(df_train[['elo_diff']], df_train['actual_win'])
model_cmp = LogisticRegression().fit(df_train[['comp_diff']], df_train['actual_win'])

# Historical Accuracy for Weights
acc_hyb = accuracy_score(df_train['actual_win'], model_hyb.predict(df_train[['hybrid_diff']]))
acc_elo = accuracy_score(df_train['actual_win'], model_elo.predict(df_train[['elo_diff']]))
acc_cmp = accuracy_score(df_train['actual_win'], model_cmp.predict(df_train[['comp_diff']]))

print(f"Accuracy -> Hybrid: {acc_hyb:.2%}, Elo: {acc_elo:.2%}, Composite: {acc_cmp:.2%}")

# 4. COMPUTE MATCHUP RESULTS
results = []
for _, row in df_comp.iterrows():
    h, a = row['home_team'], row['away_team']
    
    h_diff = hybrid_map.get(h, 0) - hybrid_map.get(a, 0)
    e_diff = elo_map.get(h, 0) - elo_map.get(a, 0)
    c_diff = comp_map.get(h, 0) - comp_map.get(a, 0)
    
    p_hyb = model_hyb.predict_proba(pd.DataFrame([[h_diff]], columns=['hybrid_diff']))[0][1]
    p_elo = model_elo.predict_proba(pd.DataFrame([[e_diff]], columns=['elo_diff']))[0][1]
    p_cmp = model_cmp.predict_proba(pd.DataFrame([[c_diff]], columns=['comp_diff']))[0][1]
    
    # Weighted Consensus Calculation
    total_acc = acc_hyb + acc_elo + acc_cmp
    final_prob = (p_hyb * acc_hyb + p_elo * acc_elo + p_cmp * acc_cmp) / total_acc
    final_winner = h if final_prob > 0.5 else a
    
    # This tuple must match the ORDER of the UPDATE statement below
    results.append((
        hybrid_map.get(h, 0), hybrid_map.get(a, 0), round(p_hyb * 100, 2),
        elo_map.get(h, 0), elo_map.get(a, 0), round(p_elo * 100, 2),
        comp_map.get(h, 0), comp_map.get(a, 0), round(p_cmp * 100, 2),
        round(final_prob * 100, 2), final_winner,
        h, a
    ))

# 5. DB WRITEBACK
cursor = conn.cursor()

# Step A: Explicitly create columns one by one
columns_needed = [
    ("home_hybrid_score", "REAL"),
    ("away_hybrid_score", "REAL"),
    ("hybrid_win_prob", "REAL"),
    ("home_elo_score", "REAL"),
    ("away_elo_score", "REAL"),
    ("elo_win_prob", "REAL"),
    ("home_composite_score", "REAL"),
    ("away_composite_score", "REAL"),
    ("composite_win_prob", "REAL"),
    ("final_win_prob", "REAL"),
    ("final_recommendation", "TEXT")
]

for col_name, col_type in columns_needed:
    try:
        cursor.execute(f"ALTER TABLE Matchups ADD COLUMN {col_name} {col_type}")
    except sqlite3.OperationalError:
        pass # Already exists

# Step B: Batch Update
update_sql = """
    UPDATE Matchups SET 
        home_hybrid_score = ?, 
        away_hybrid_score = ?, 
        hybrid_win_prob = ?,
        home_elo_score = ?, 
        away_elo_score = ?, 
        elo_win_prob = ?,
        home_composite_score = ?, 
        away_composite_score = ?, 
        composite_win_prob = ?,
        final_win_prob = ?, 
        final_recommendation = ?
    WHERE home_team = ? AND away_team = ?
"""

cursor.executemany(update_sql, results)
conn.commit()
conn.close()
print("🏆 Success! Matchups table updated with weighted recommendations.")