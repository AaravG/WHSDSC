import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import sqlite3
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

cat_cols = [
  "home_team","away_team","home_off_line","home_def_pairing",
  "away_off_line","away_def_pairing","home_goalie","away_goalie"
]

num_cols = [
  "went_ot","toi","home_assists","home_shots","home_xg","home_max_xg",
  "away_assists","away_shots","away_xg","away_max_xg",
  "home_penalties_committed","home_penalty_minutes",
  "away_penalties_committed","away_penalty_minutes"
]



conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")
df = pd.read_sql("SELECT * FROM kNN", conn)
conn.close()

df = df.sample(frac=1)
train = df.iloc[:int(0.6*len(df))]
valid = df.iloc[int(0.6*len(df)):int(0.8*len(df))]
test  = df.iloc[int(0.8*len(df)):]

def scale_dataset(df, oversample=False):
    X = df[cat_cols + num_cols]
    y = df["home_win"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    X = preprocessor.fit_transform(X)

    if oversample:
        X, y = RandomOverSampler().fit_resample(X, y)

    return X, y

X_train, y_train = scale_dataset(train, oversample=True)
X_valid, y_valid = scale_dataset(valid, oversample=False)
X_test,  y_test  = scale_dataset(test, oversample=False)

li = []

for i in range(1, 26):
    knn_model = KNeighborsClassifier(n_neighbors=i)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_valid)
    print(f"Neighbors: {i}", end="\n --------------- \n")
    print(classification_report(y_valid, y_pred))
    print("------------------\n\n")
    li.append((i, classification_report(y_valid, y_pred, output_dict=True)["accuracy"]))

        


# 1. Prepare the data
x = np.array([item[0] for item in li])
y = np.array([item[1] for item in li])

# 1. Calculate the Moving Average (the "Mid-line")
window_size = 10  # Increase this for more smoothing, decrease to follow closer
y_pd = pd.Series(y)
y_moving_avg = y_pd.rolling(window=window_size, center=True).mean()

# 2. Setup Plot
plt.figure(figsize=(12, 6))

# The Original Jagged Line (Light Blue)
plt.plot(x, y, color='skyblue', alpha=0.8, label='Raw Accuracy')

# The "Mid-line" (Dark Blue - Moving Average)
# This stays close to the data but removes the tiny spikes
plt.plot(x, y_moving_avg, color='red', linewidth=2, label=f'Moving Average (n={window_size})')

# 3. Styling
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('kNN Accuracy: Raw vs. Moving Average Mid-line')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()