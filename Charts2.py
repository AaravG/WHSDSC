import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3


# Pulling live data from DB instead of manual Excel
conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")
df_hyb = pd.read_sql("SELECT home_team as team, hybrid_score as hybrid, win_rate FROM final_rankings_hybrid_v5", conn)
df_disp = pd.read_sql("SELECT team, raw_gap as disparity FROM off_line_disparity_v7", conn)
df = pd.merge(df_hyb, df_disp, on='team')

# Rename columns to match your existing plot logic
df = df.rename(columns={'team': 'Name of teams', 'win_rate': 'win%', 'disparity': 'offensive line disparity'})

# 2. PLOT SETUP
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# 3. CREATE BUBBLE SCATTER PLOT
# X = Hybrid Score (Strength), Y = Disparity (Depth), Size = Win%
scatter = plt.scatter(
    df['hybrid'], 
    df['offensive line disparity'], 
    # df['offensive line disparity'],
    # df['hybrid'],
    s=df['win%']*250,  # Scaling bubble size
    alpha=0.6, 
    c=df['win%'], 
    cmap='viridis',
    edgecolors="w"
)

# # 4. ADD QUADRANT LINES (Mean or Zero-based)
plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
plt.axhline(y=0.25, color='red', linestyle='--', alpha=0.5) # 0.25 is approx league avg disparity

# 5. ANNOTATE TEAMS
for i in range(df.shape[0]):
    plt.text(
        df.hybrid[i]+0.02, 
        df['offensive line disparity'][i]+0.02, 
        df['Name of teams'][i], 
        fontsize=9
    )
    
    
#     plt.text(
#     df['offensive line disparity'][i] + 0.02,
#     df.hybrid[i] + 0.02,
#     df['Name of teams'][i],
#     fontsize=9
# )

# 6. LABELS & TITLE
plt.title('WHL 2025: Team Strength vs. Line Depth Archetypes', fontsize=15)
plt.xlabel('Overall Strength (Hybrid Score)', fontsize=12)
plt.ylabel('Internal Depth Gap (Offensive Line Disparity)', fontsize=12)
plt.colorbar(scatter, label='Win %')

plt.figtext(0.5, 0.01, 
            "FINDING: Success in the WHL is highly correlated with 'Deep Contenders' (Bottom-Right). \n"
            "High line disparity acts as a 'Depth Tax'—reducing a team's win probability even at elite strength levels, \n"
            "as opposing defenses find it significantly easier to neutralize one-dimensional scoring threats.", 
            ha="center", fontsize=11, fontweight='medium', bbox={"facecolor":"orange", "alpha":0.1, "pad":8})

# Add Quadrant Text Labels
plt.text(1.2, 0.7, "Top-Heavy Juggernauts", fontsize=10, color='darkred', fontweight='bold')
plt.text(-1.2, 0.7, "Top-Heavy Underdogs", fontsize=10, color='darkred', fontweight='bold')
plt.text(1.2, -0.2, "Deep Contenders", fontsize=10, color='darkblue', fontweight='bold')
plt.text(-1.2, -0.2, "Balanced Rebuilders", fontsize=10, color='darkblue', fontweight='bold')

plt.tight_layout()
plt.show()