import pandas as pd
import numpy as np
import sqlite3
import statsmodels.api as sm

# -------------------------------------------------
# 1. CONNECT
# -------------------------------------------------
conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")
df = pd.read_sql("SELECT * FROM data", conn)

# -------------------------------------------------
# 2. REMOVE EMPTY NET
# -------------------------------------------------
df = df[
    (df["home_goalie"] != "empty_net_line") &
    (df["away_goalie"] != "empty_net_line")
].copy()

# -------------------------------------------------
# 3. RESHAPE TO LONG FORMAT
# -------------------------------------------------
home_df = pd.DataFrame({
    "team": df["home_team"],
    "off_line": df["home_off_line"],
    "def_line": df["away_def_pairing"],
    "xg": df["home_xg"],
    "toi": df["toi"]
})

away_df = pd.DataFrame({
    "team": df["away_team"],
    "off_line": df["away_off_line"],
    "def_line": df["home_def_pairing"],
    "xg": df["away_xg"],
    "toi": df["toi"]
})

long_df = pd.concat([home_df, away_df], ignore_index=True)

# -------------------------------------------------
# 4. CLEAN
# -------------------------------------------------
long_df = long_df[
    ~long_df["off_line"].isin(["off_PP_up", "off_empty_net_line"])
].copy()

long_df["xg"] = pd.to_numeric(long_df["xg"], errors="coerce")
long_df["toi"] = pd.to_numeric(long_df["toi"], errors="coerce")

long_df = long_df.dropna(subset=["xg", "toi"])
long_df = long_df[long_df["toi"] > 0]

# xG per 60
long_df["xg_rate"] = (long_df["xg"] / long_df["toi"]) * 60

# -------------------------------------------------
# 5. BUILD IDS
# -------------------------------------------------
long_df["off_id"] = (
    "off_" +
    long_df["team"].astype(str) +
    "__" +
    long_df["off_line"].astype(str)
)

long_df["def_id"] = "def_" + long_df["def_line"].astype(str)

# -------------------------------------------------
# 6. DESIGN MATRIX (FORCE NUMERIC)
# -------------------------------------------------
off_dummies = pd.get_dummies(long_df["off_id"], drop_first=True)
def_dummies = pd.get_dummies(long_df["def_id"], drop_first=True)

X = pd.concat([off_dummies, def_dummies], axis=1)

X = X.astype(float)        # 🔥 CRITICAL FIX
X = sm.add_constant(X)

y = long_df["xg_rate"].astype(float)
weights = long_df["toi"].astype(float)

# -------------------------------------------------
# 7. FIT MODEL
# -------------------------------------------------
model = sm.WLS(y, X, weights=weights).fit()

# -------------------------------------------------
# 8. EXTRACT RESULTS
# -------------------------------------------------
teams = long_df["team"].unique()
results = []

for team in teams:

    first_name = f"off_{team}__first_off"
    second_name = f"off_{team}__second_off"

    beta_first = model.params.get(first_name, 0.0)
    beta_second = model.params.get(second_name, 0.0)

    raw_gap = beta_first - beta_second

    if first_name in model.params and second_name in model.params:
        t_test = model.t_test(f"{first_name} - {second_name} = 0")
        gap = float(t_test.effect.item())
        t_stat = float(t_test.tvalue.item())
        p_val = float(t_test.pvalue.item())
        weighted_gap = abs(gap) * abs(t_stat)
    else:
        t_stat = 0.0
        p_val = 1.0
        weighted_gap = 0.0

    results.append({
        "team": team,
        "first_line_score": beta_first,
        "second_line_score": beta_second,
        "disparity": raw_gap,
        "t_stat_gap": t_stat,
        "p_value_gap": p_val,
        "weighted_disparity": weighted_gap
    })

team_df = pd.DataFrame(results)

# -------------------------------------------------
# 9. RANK
# -------------------------------------------------
team_df["rank"] = (
    team_df["disparity"].abs()
    .rank(ascending=False, method="min")
    .astype(int)
)

team_df["weighted_rank"] = (
    team_df["weighted_disparity"]
    .rank(ascending=False, method="min")
    .astype(int)
)

team_df = team_df.sort_values("weighted_rank")

# -------------------------------------------------
# 10. SAVE
# -------------------------------------------------
team_df.to_sql(
    "off_line_disparity",
    conn,
    if_exists="replace",
    index=False
)

print("Saved table: off_line_disparity")
print(team_df)
print(model.summary())

conn.close()