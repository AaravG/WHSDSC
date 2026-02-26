import pandas as pd
import numpy as np
import sqlite3
import statsmodels.api as sm

# 1. CONNECT & LOAD
conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")
df = pd.read_sql("SELECT * FROM data", conn)

# 2. CLEAN & RESHAPE
df = df[(df["home_goalie"] != "empty_net_line") & (df["away_goalie"] != "empty_net_line")].copy()

home_df = pd.DataFrame({
    "team": df["home_team"], "off_line": df["home_off_line"],
    "def_line": df["away_def_pairing"], "xg": df["home_xg"], "toi": df["toi"]
})
away_df = pd.DataFrame({
    "team": df["away_team"], "off_line": df["away_off_line"],
    "def_line": df["home_def_pairing"], "xg": df["away_xg"], "toi": df["toi"]
})

long_df = pd.concat([home_df, away_df], ignore_index=True)
long_df = long_df[~long_df["off_line"].isin(["off_PP_up", "off_empty_net_line"])].copy()

# 3. RE-CENTERING (The "Negative Value" Fix)
long_df["xg_rate"] = (long_df["xg"].astype(float) / long_df["toi"].astype(float)) * 60
long_df = long_df.dropna(subset=["xg_rate"])
long_df = long_df[long_df["toi"] > 0]

# Calculate league average to make coefficients intuitive (relative to mean)
league_avg_xg = np.average(long_df["xg_rate"], weights=long_df["toi"])
long_df["xg_rel"] = long_df["xg_rate"] - league_avg_xg

# 4. DESIGN MATRIX
long_df["off_id"] = "off_" + long_df["team"] + "__" + long_df["off_line"]
long_df["def_id"] = "def_" + long_df["def_line"]

off_dummies = pd.get_dummies(long_df["off_id"]).astype(float)
def_dummies = pd.get_dummies(long_df["def_id"]).astype(float)

X = pd.concat([off_dummies, def_dummies], axis=1)
y = long_df["xg_rel"]
weights = long_df["toi"]

# 5. FIT MODEL (Adding a tiny bit of L2 penalty to solve the Singularity/Note [2] issue)
# We use standard WLS but can use a 'Ridge' approach if Note [2] persists
model = sm.WLS(y, X, weights=weights).fit()

# 6. EXTRACT RESULTS
results = []
teams = long_df["team"].unique()

for team in teams:
    f_name, s_name = f"off_{team}__first_off", f"off_{team}__second_off"
    
    if f_name in model.params and s_name in model.params:
        # Statistical test for the gap
        t_test = model.t_test(f"{f_name} - {s_name} = 0")
        
        # Absolute Score = League Average + Coefficient
        # This makes the numbers look like real xG/60 again
        score_l1 = league_avg_xg + model.params[f_name]
        score_l2 = league_avg_xg + model.params[s_name]
        
        gap = t_test.effect.item()
        t_stat = t_test.tvalue.item()
        p_val = t_test.pvalue.item()
        
        results.append({
            "team": team,
            "L1_Adj_xG60": round(score_l1 * 60, 3), # Converting to Hourly rate
            "L2_Adj_xG60": round(score_l2 * 60, 3),
            "raw_gap": round(gap * 60, 3),
            "t_stat": round(t_stat, 2),
            "p_val": round(p_val, 4),
            "reliability_score": round(abs(gap * 60) * abs(t_stat), 2)
        })

team_df = pd.DataFrame(results).sort_values("reliability_score", ascending=False)
team_df["rank"] = range(1, len(team_df) + 1)

# 7. OUTPUTS
print("\n--- RE-CENTERED MODEL SUMMARY ---")
print(model.summary())

print("\n--- FINAL TEAM DISPARITY RANKINGS ---")
print(team_df[["rank", "team", "L1_Adj_xG60", "L2_Adj_xG60", "raw_gap", "p_val"]].to_string(index=False))

# 8. SAVE
team_df.to_sql("off_line_disparity_v2", conn, if_exists="replace", index=False)
conn.close()