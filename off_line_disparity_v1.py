import pandas as pd
import numpy as np
import sqlite3
import statsmodels.api as sm

# ==========================================
# 1. LOAD DATA
# ==========================================
DB_PATH = r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM data", conn)
conn.close()

# ==========================================
# 2. REMOVE EMPTY NET + SPECIAL TEAMS
# ==========================================
df = df[
    (df["home_goalie"] != "empty_net_line") &
    (df["away_goalie"] != "empty_net_line")
]

df = df[
    (~df["home_off_line"].isin(["PP_up", "empty_net_line"])) &
    (~df["away_off_line"].isin(["PP_up", "empty_net_line"]))
]

# ==========================================
# 3. CLEAN DATA
# ==========================================
df = df.dropna(subset=[
    "home_off_line", "away_off_line",
    "home_def_pairing", "away_def_pairing",
    "home_xg", "away_xg", "toi",
    "home_team", "away_team"
])

# ==========================================
# 4. BUILD OFFENSIVE DATA (HOME + AWAY)
# ==========================================
home_df = pd.DataFrame({
    "team": df["home_team"],
    "off_line": df["home_off_line"],
    "def_pairing": df["away_def_pairing"],
    "xg": df["home_xg"],
    "toi": df["toi"]
})

away_df = pd.DataFrame({
    "team": df["away_team"],
    "off_line": df["away_off_line"],
    "def_pairing": df["home_def_pairing"],
    "xg": df["away_xg"],
    "toi": df["toi"]
})

off_df = pd.concat([home_df, away_df], ignore_index=True)

off_df["xg_rate"] = off_df["xg"] / off_df["toi"]
off_df = off_df.replace([np.inf, -np.inf], np.nan).dropna()

# ==========================================
# 5. TEAM-SPECIFIC LINE IDENTIFIER
# ==========================================
off_df["team_line"] = off_df["team"] + "__" + off_df["off_line"]

# ==========================================
# 6. FIXED EFFECTS REGRESSION
# ==========================================
off_dummies = pd.get_dummies(
    off_df["team_line"],
    prefix="off",
    drop_first=True
)

def_dummies = pd.get_dummies(
    off_df["def_pairing"],
    prefix="def",
    drop_first=True
)

X = pd.concat([off_dummies, def_dummies], axis=1)
X = sm.add_constant(X).astype(float)

y = off_df["xg_rate"].astype(float)
weights = off_df["toi"].astype(float)

model = sm.WLS(y, X, weights=weights).fit()

# ==========================================
# 7. EXTRACT OFFENSIVE LINE SCORES
# ==========================================
line_strength = model.params.filter(like="off_")

line_strength_df = pd.DataFrame({
    "team_line": line_strength.index.str.replace("off_", "", regex=False),
    "score": line_strength.values
})

# Split team and line safely
line_strength_df[["team", "off_line"]] = (
    line_strength_df["team_line"]
    .str.split("__", expand=True)
)

# ==========================================
# 8. CLASSIFY FIRST / SECOND LINES
# ==========================================
def classify_line(name):
    name = str(name).lower()
    if "first" in name:
        return "first"
    elif "second" in name:
        return "second"
    else:
        return None

line_strength_df["line_type"] = (
    line_strength_df["off_line"].apply(classify_line)
)

line_strength_df = line_strength_df.dropna(subset=["line_type"])

# ==========================================
# 9. TEAM-LEVEL DISPARITY
# ==========================================
team_df = line_strength_df.pivot_table(
    index="team",
    columns="line_type",
    values="score",
    aggfunc="mean"
).reset_index()

# Ensure both columns exist
if "first" not in team_df.columns:
    team_df["first"] = np.nan
if "second" not in team_df.columns:
    team_df["second"] = np.nan

team_df = team_df.rename(columns={
    "first": "first_line_score",
    "second": "second_line_score"
})

team_df = team_df.dropna(
    subset=["first_line_score", "second_line_score"]
)

team_df["disparity"] = (
    team_df["first_line_score"] -
    team_df["second_line_score"]
)

team_df = team_df.sort_values(
    "disparity",
    ascending=False
)

team_df["rank"] = range(1, len(team_df) + 1)



# -------------------------------------------------
# ADD SIGNIFICANCE-WEIGHTED DISPARITY
# -------------------------------------------------

weighted_list = []

for team in team_df["team"]:
    
    first_param = f"off_{team}__first_off"
    second_param = f"off_{team}__second_off"
    
    if first_param in model.params and second_param in model.params:
        
        beta1 = model.params[first_param]
        beta2 = model.params[second_param]
        
        se1 = model.bse[first_param]
        se2 = model.bse[second_param]
        
        # Gap
        gap = beta1 - beta2
        
        # Conservative SE of difference (ignoring covariance)
        se_gap = (se1**2 + se2**2) ** 0.5
        
        if se_gap > 0:
            t_value = gap / se_gap
        else:
            t_value = 0
        
        weighted_disparity = abs(gap) * abs(t_value)
        
    else:
        weighted_disparity = 0
    
    weighted_list.append(weighted_disparity)

team_df["weighted_disparity"] = weighted_list

# Re-rank based on weighted metric
team_df["weighted_rank"] = (
    team_df["weighted_disparity"]
    .rank(ascending=False, method="min")
    .astype(int)
)


# ==========================================
# 10. SAVE TO DATABASE
# ==========================================
conn = sqlite3.connect(DB_PATH)

team_df.to_sql(
    "off_line_disparity",
    conn,
    if_exists="replace",
    index=False
)

conn.close()

print("Saved table: off_line_disparity")
print(team_df)
print(model.summary())