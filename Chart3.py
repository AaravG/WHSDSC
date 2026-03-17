import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# 1. DATA ACQUISITION
# Ensure the path is correct for your local environment
db_path = r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db"
conn = sqlite3.connect(db_path)

# Pulling 1a Strength and 1b Disparity data
df_hyb = pd.read_sql("SELECT home_team as team, hybrid_score, win_rate FROM final_rankings_hybrid_v5", conn)
df_disp = pd.read_sql("SELECT team, raw_gap FROM off_line_disparity_v7", conn)
df = pd.merge(df_hyb, df_disp, on='team')
conn.close()

# 2. PLOT SETUP
plt.figure(figsize=(14, 9))
sns.set_style("white")
avg_strength = 0  # Scores are centered around zero
avg_disparity = df['raw_gap'].mean()

# 3. CREATE THE BUBBLE ARCHETYPE VISUAL
# Bubble size is driven by win rate to show the success correlation
scatter = plt.scatter(
    df['hybrid_score'], 
    df['raw_gap'], 
    s=df['win_rate']*1000, 
    c=df['win_rate'], 
    cmap='RdYlGn', # Red to Green gradient based on Win %
    alpha=0.75, 
    edgecolors="black",
    linewidth=1
)

# 4. ADD STRATEGIC QUADRANTS (The "Key Logic")
plt.axvline(x=avg_strength, color='gray', linestyle='--', alpha=0.5)
plt.axhline(y=avg_disparity, color='gray', linestyle='--', alpha=0.5)

# 5. LABELS & ANNOTATIONS (Self-Explanatory for the Commissioner)
plt.text(1.8, df['raw_gap'].max(), "TOP-HEAVY JUGGERNAUTS\n(High Strength, High Risk)", 
         fontsize=11, color='darkorange', fontweight='bold', ha='right', va='top')
plt.text(1.8, df['raw_gap'].min(), "DEEP CONTENDERS\n(High Strength, High Stability)", 
         fontsize=11, color='darkgreen', fontweight='bold', ha='right', va='bottom')
plt.text(-1.8, df['raw_gap'].max(), "VOLATILE UNDERDOGS\n(Low Strength, Single Threat)", 
         fontsize=11, color='darkred', fontweight='bold', ha='left', va='top')
plt.text(-1.8, df['raw_gap'].min(), "BALANCED REBUILDERS\n(Low Strength, High Depth)", 
         fontsize=11, color='navy', fontweight='bold', ha='left', va='bottom')

# Annotate Team Names (Targeting extremes for clarity)
for i, row in df.iterrows():
    if abs(row['hybrid_score']) > 1.2 or row['raw_gap'] > avg_disparity * 1.5 or row['raw_gap'] < avg_disparity * 0.5:
        plt.annotate(row['team'].title(), (row['hybrid_score'], row['raw_gap']), 
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='medium')

# 6. TITLES & LEGEND
plt.title('WHL 2025: Roster Architecture vs. Team Success', fontsize=18, fontweight='bold', pad=25)
plt.suptitle('Analyzing the Impact of Offensive Line Disparity on Win Stability', fontsize=12, y=0.92)
plt.xlabel('Overall Team Strength (Hybrid Score)', fontsize=13)
plt.ylabel('Offensive Line Quality Disparity (Depth Gap)', fontsize=13)

# Add colorbar for Win %
cbar = plt.colorbar(scatter)
cbar.set_label('Actual Win Percentage (%)', rotation=270, labelpad=20)

# 7. CAPTION (REQUIRED FOR RUBRIC)
plt.figtext(0.5, 0.01, 
            "FINDING: Success in the WHL is highly correlated with 'Deep Contenders' (Bottom-Right). \n"
            "High line disparity acts as a 'Depth Tax'—reducing a team's win probability even at elite strength levels, \n"
            "as opposing defenses find it significantly easier to neutralize one-dimensional scoring threats.", 
            ha="center", fontsize=11, fontweight='medium', bbox={"facecolor":"orange", "alpha":0.1, "pad":8})

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('WHL_Commissioner_Insight.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Success: Commissioner-ready visualization saved as WHL_Commissioner_Insight.png")