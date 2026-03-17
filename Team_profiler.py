import pandas as pd
import sqlite3

# 1. Load the data
conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")
df = pd.read_sql("SELECT * FROM final_rankings_hybrid_v5", conn)
conn.close()

def generate_report(row):
    team = row['home_team']
    rank = int(row['rank'])
    win_rate = row['win_rate'] * 100
    
    # Identify the biggest driver (highest impact)
    impact_cols = [c for c in df.columns if c.startswith('impact_')]
    biggest_driver_col = df[impact_cols].loc[row.name].idxmax()
    biggest_driver_val = df[impact_cols].loc[row.name].max()
    
    # Identify the biggest weakness (lowest impact)
    biggest_weakness_col = df[impact_cols].loc[row.name].idxmin()
    biggest_weakness_val = df[impact_cols].loc[row.name].min()

    # Determine "Luck" (Elo vs Efficiency)
    # If Elo is way higher than Efficiency, they are 'Legacy/Lucky'
    # If Efficiency is way higher than Elo, they are 'Rising Stars'
    diff = row['elo_scaled'] - row['efficiency_scaled']
    if diff > 0.5:
        status = "Legacy Powerhouse (Winning on Reputation/Luck)"
    elif diff < -0.5:
        status = "Under-the-Radar (Playing better than their record)"
    else:
        status = "Balanced (Rank is exactly where it should be)"

    report = f"""
--- SCOUTING REPORT: {team.upper()} ---
Rank: #{rank} | Win Rate: {win_rate:.1f}%
Status: {status}

[STRENGTH] 
The model loves their {biggest_driver_col.replace('impact_', '')}. 
It adds +{biggest_driver_val:.3f} to their win probability.

[WEAKNESS] 
They are struggling with {biggest_weakness_col.replace('impact_', '')}. 
It drags them down by {biggest_weakness_val:.3f}.

[HYBRID INSIGHT]
Elo Scaled: {row['elo_scaled']:.2f} vs Efficiency Scaled: {row['efficiency_scaled']:.2f}
Hybrid Score: {row['hybrid_score']:.3f}
    """
    return report

# Print the top 10 teams
print("GENERATING WHL ANALYTICS REPORTS...")
for i in range(32):
    print(generate_report(df.iloc[i]))
    
    
    
with open("Team_Profiler.txt", "a") as file:
        for i in range(32):
            file.write(generate_report(df.iloc[i]))
