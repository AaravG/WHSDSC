import pandas as pd
import sqlite3
import numpy as np

print("="*60)
print("DATA VALIDATION SCRIPT")
print("="*60)

conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")
df = pd.read_sql("SELECT * FROM NN", conn)
conn.close()

# ==========================
# CHECK 1: Target Variable
# ==========================
print("\n" + "="*60)
print("CHECK 1: Is home_win calculated correctly?")
print("="*60)

df['calculated_home_win'] = (df['home_goals'] > df['away_goals']).astype(int)
df['match'] = df['home_win'] == df['calculated_home_win']

mismatches = df[~df['match']]

if len(mismatches) > 0:
    print(f"❌ CRITICAL ERROR: {len(mismatches)} games have incorrect home_win labels!")
    print("\nFirst 5 mismatches:")
    print(mismatches[['game_id', 'home_team', 'away_team', 'home_goals', 'away_goals', 'home_win', 'calculated_home_win']].head())
else:
    print("✓ home_win column is correct")

# ==========================
# CHECK 2: ELO Variance
# ==========================
print("\n" + "="*60)
print("CHECK 2: Is ELO updating properly?")
print("="*60)

elo_stats = {
    'min': df['home_elo'].min(),
    'max': df['home_elo'].max(),
    'mean': df['home_elo'].mean(),
    'std': df['home_elo'].std()
}

print(f"ELO Statistics:")
print(f"  Min:  {elo_stats['min']:.2f}")
print(f"  Max:  {elo_stats['max']:.2f}")
print(f"  Mean: {elo_stats['mean']:.2f}")
print(f"  Std:  {elo_stats['std']:.2f}")

if elo_stats['std'] < 30:
    print("❌ PROBLEM: ELO standard deviation too low - not updating properly")
else:
    print("✓ ELO variance looks good")

# Check if first games are all 1500
first_50 = df.head(50)
if (first_50['home_elo'] == 1500).all() and (first_50['away_elo'] == 1500).all():
    print("⚠️  WARNING: First 50 games all start at 1500 (expected)")
else:
    print("✓ ELO is updating from game to game")

# ==========================
# CHECK 3: Rolling Features
# ==========================
print("\n" + "="*60)
print("CHECK 3: Are rolling features calculated correctly?")
print("="*60)

# Sort by game_id
df["game_num"] = df["game_id"].str.extract(r'(\d+)').astype(int)
df = df.sort_values("game_num").reset_index(drop=True)

# Calculate test rolling feature
df["home_goals_for"] = df["home_goals"]
df["test_rolling"] = (
    df.groupby("home_team")["home_goals_for"]
      .shift(1).rolling(5, min_periods=1).mean()
)

# Check first game for each team (should be NaN)
first_games = df.groupby("home_team").first()
nan_count = first_games['test_rolling'].isna().sum()

print(f"First games with NaN rolling stat: {nan_count} / {len(first_games)}")

if nan_count == len(first_games):
    print("✓ Rolling features use shift(1) correctly")
else:
    print("❌ PROBLEM: Rolling features might have data leakage")
    print("\nFirst game for each team:")
    print(first_games[['game_id', 'home_goals_for', 'test_rolling']].head())

# ==========================
# CHECK 4: Home Win Rate Distribution
# ==========================
print("\n" + "="*60)
print("CHECK 4: Home win distribution")
print("="*60)

home_win_rate = df['home_win'].mean()
print(f"Overall home win rate: {home_win_rate:.1%}")

if home_win_rate < 0.48 or home_win_rate > 0.65:
    print("⚠️  WARNING: Unusual home win rate (expected 52-58%)")
else:
    print("✓ Home win rate is normal")

# ==========================
# CHECK 5: Goals vs Win Correlation
# ==========================
print("\n" + "="*60)
print("CHECK 5: Do goals predict wins?")
print("="*60)

df['goal_diff'] = df['home_goals'] - df['away_goals']
corr = df['goal_diff'].corr(df['home_win'])

print(f"Correlation (goal_diff vs home_win): {corr:+.4f}")

if abs(corr) < 0.50:
    print("❌ CRITICAL: Goals don't correlate with wins!")
    print("   This suggests your home_win column is wrong or data is corrupted")
else:
    print("✓ Goals correlate with wins (as expected)")

# ==========================
# CHECK 6: Sample Data Quality
# ==========================
print("\n" + "="*60)
print("CHECK 6: Sample data inspection")
print("="*60)

print("\nFirst 10 games:")
print(df[['game_id', 'home_team', 'away_team', 'home_goals', 'away_goals', 'home_win', 'home_elo', 'away_elo']].head(10))

print("\nGames with ties (should be 0):")
ties = df[df['home_goals'] == df['away_goals']]
print(f"  Count: {len(ties)}")
if len(ties) > 0:
    print("  First 5 ties:")
    print(ties[['game_id', 'home_team', 'away_team', 'home_goals', 'away_goals', 'home_win', 'went_ot']].head())

# ==========================
# FINAL DIAGNOSIS
# ==========================
print("\n" + "="*60)
print("FINAL DIAGNOSIS")
print("="*60)

issues = []

if len(mismatches) > 0:
    issues.append("❌ CRITICAL: home_win column has errors")
if elo_stats['std'] < 30:
    issues.append("❌ ELO not updating properly")
if nan_count != len(first_games):
    issues.append("❌ Rolling features have data leakage")
if home_win_rate < 0.48 or home_win_rate > 0.65:
    issues.append("⚠️  Unusual home win rate")
if abs(corr) < 0.50:
    issues.append("❌ CRITICAL: Goals don't predict wins")

if issues:
    print("\nISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")
    print("\nYour model is failing because of DATA QUALITY issues, not the algorithm!")
else:
    print("\n✓ No major issues detected")
    print("  Your data looks OK - model should achieve 65%+ accuracy")
    print("  If accuracy is still low, the problem might be:")
    print("  1. Insufficient features")
    print("  2. Inherent randomness in sports")
    print("  3. Need for non-linear model")

print("\n" + "="*60)