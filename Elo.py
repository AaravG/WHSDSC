import sqlite3
import pandas as pd
import numpy as np

# ==========================
# CONFIG
# ==========================
DB_PATH = r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db"
TABLE_NAME = "NN"
K = 30
INITIAL_RATING = 1500

# ==========================
# CONNECT & LOAD
# ==========================
conn = sqlite3.connect(DB_PATH)

df = pd.read_sql(
    f"SELECT rowid, home_team, away_team, home_win FROM {TABLE_NAME} ORDER BY rowid",
    conn
)

# ==========================
# ENSURE COLUMNS EXIST
# ==========================
try:
    conn.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN home_elo REAL;")
except:
    pass

try:
    conn.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN away_elo REAL;")
except:
    pass

# Ensure columns exist
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(NN);")
cols = [row[1] for row in cursor.fetchall()]

if "home_elo" not in cols:
    cursor.execute("ALTER TABLE NN ADD COLUMN home_elo REAL;")

if "away_elo" not in cols:
    cursor.execute("ALTER TABLE NN ADD COLUMN away_elo REAL;")

conn.commit()

# ==========================
# INITIALIZE ELO
# ==========================
teams = set(df["home_team"]).union(set(df["away_team"]))
elo = {team: INITIAL_RATING for team in teams}

home_elos = []
away_elos = []

# ==========================
# SEQUENTIAL ELO CALCULATION
# ==========================
for _, row in df.iterrows():
    home = row["home_team"]
    away = row["away_team"]
    result = row["home_win"]  # 1 if home win, 0 if away win

    Rh = elo[home]
    Ra = elo[away]

    # Store PRE-game ratings
    home_elos.append(Rh)
    away_elos.append(Ra)

    # Expected score
    Eh = 1 / (1 + 10 ** ((Ra - Rh) / 400))

    # Update ratings
    elo[home] = Rh + K * (result - Eh)
    elo[away] = Ra + K * ((1 - result) - (1 - Eh))

df["home_elo"] = home_elos
df["away_elo"] = away_elos

# ==========================
# WRITE BACK TO DATABASE
# ==========================
cursor = conn.cursor()

for _, row in df.iterrows():
    cursor.execute(
        f"""
        UPDATE {TABLE_NAME}
        SET home_elo = ?,
            away_elo = ?
        WHERE rowid = ?
        """,
        (row["home_elo"], row["away_elo"], row["rowid"])
    )

conn.commit()
conn.close()

print("✓ Elo columns computed and saved successfully.")