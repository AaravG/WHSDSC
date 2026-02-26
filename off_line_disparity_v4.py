import pandas as pd
import numpy as np
import sqlite3
from sklearn.linear_model import RidgeCV

# 1. CONNECT & LOAD
conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")
df = pd.read_sql("SELECT * FROM data", conn)

# 2. EMPTY NET FILTERING
df = df[(df['away_goalie'] != 'Empty') & (df['home_goalie'] != 'Empty')]

for col in ['home_off_line', 'away_off_line']:
    df = df[~df[col].str.contains('empty|PP', case=False, na=False)]

# 3. RESHAPE
h_df = pd.DataFrame({
    "team": df["home_team"],
    "off_line": df["home_off_line"],
    "def_line": df["away_def_pairing"],
    "xg": df["home_xg"],
    "toi": df["toi"]
})

a_df = pd.DataFrame({
    "team": df["away_team"],
    "off_line": df["away_off_line"],
    "def_line": df["home_def_pairing"],
    "xg": df["away_xg"],
    "toi": df["toi"]
})

long_df = pd.concat([h_df, a_df], ignore_index=True)

# 4. FILTER & RATE
long_df = long_df[
    long_df['off_line'].str.contains('1|2|first|second', case=False, na=False)
]

long_df["xg_rate"] = (
    long_df["xg"].astype(float) /
    long_df["toi"].astype(float)
) * 3600

long_df = long_df.replace([np.inf, -np.inf], np.nan)
long_df = long_df.dropna(subset=["xg_rate"])

# 5. RIDGECV MODEL (no drop_first)
long_df["off_id"] = "off_" + long_df["team"] + "_" + long_df["off_line"]

X_off = pd.get_dummies(long_df["off_id"])
X_def = pd.get_dummies(long_df["def_line"], prefix="def")
X = pd.concat([X_off, X_def], axis=1)

y = long_df["xg_rate"]
weights = long_df["toi"]

ridge_cv = RidgeCV(
    alphas=[0.1, 1.0, 10.0, 100.0],
    scoring='neg_mean_squared_error',
    cv=5
)

ridge_cv.fit(X, y, sample_weight=weights)

coeffs = dict(zip(X.columns, ridge_cv.coef_))

# 6. EXTRACT LINE GAPS
results = []
unique_teams = long_df["team"].unique()

for team in unique_teams:
    l1_cols = [
        c for c in X_off.columns
        if c.startswith(f"off_{team}_") and
        ('1' in c.lower() or 'first' in c.lower())
    ]

    l2_cols = [
        c for c in X_off.columns
        if c.startswith(f"off_{team}_") and
        ('2' in c.lower() or 'second' in c.lower())
    ]

    if not l1_cols or not l2_cols:
        continue

    l1_coef = np.mean([coeffs.get(col, 0) for col in l1_cols])
    l2_coef = np.mean([coeffs.get(col, 0) for col in l2_cols])

    raw_gap = l1_coef - l2_coef

    results.append({
        "team": team,
        "line1_xg60_adj": l1_coef,
        "line2_xg60_adj": l2_coef,
        "raw_gap": raw_gap
    })

team_df = pd.DataFrame(results).sort_values("raw_gap", ascending=False)
team_df["rank"] = range(1, len(team_df) + 1)

# 7. SAVE
team_df.to_sql("off_line_disparity_v7", conn, if_exists="replace", index=False)
conn.close()

print(f"Processed {len(team_df)} teams out of {len(unique_teams)} total.")