import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD DATA
file_path = r"E:\Aarav PC Data\Code\WHSDSC\whl_2025_phase_1b.xlsx"
df = pd.read_excel(file_path, sheet_name="1c")

# 2. PLOT SETUP
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# 3. CREATE BUBBLE SCATTER PLOT
# X = Hybrid Score (Strength), Y = Disparity (Depth), Size = Win%
scatter = plt.scatter(
    df['hybrid'], 
    df['offensive line disparity'], 
    s=df['win%']*5,  # Scaling bubble size
    alpha=0.6, 
    c=df['win%'], 
    cmap='viridis',
    edgecolors="w"
)

# 4. ADD QUADRANT LINES (Mean or Zero-based)
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

# 6. LABELS & TITLE
plt.title('WHL 2025: Team Strength vs. Line Depth Archetypes', fontsize=15)
plt.xlabel('Overall Strength (Hybrid Score)', fontsize=12)
plt.ylabel('Internal Depth Gap (Offensive Line Disparity)', fontsize=12)
plt.colorbar(scatter, label='Win %')

# Add Quadrant Text Labels
plt.text(1.2, 0.7, "Top-Heavy Juggernauts", fontsize=10, color='darkred', fontweight='bold')
plt.text(-1.2, 0.7, "Top-Heavy Underdogs", fontsize=10, color='darkred', fontweight='bold')
plt.text(1.2, -0.2, "Deep Contenders", fontsize=10, color='darkblue', fontweight='bold')
plt.text(-1.2, -0.2, "Balanced Rebuilders", fontsize=10, color='darkblue', fontweight='bold')

plt.tight_layout()
plt.show()