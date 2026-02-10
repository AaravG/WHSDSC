import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import sqlite3
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB


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

    # --- ADD THIS LINE TO FIX THE ERROR ---
    if hasattr(X, "toarray"):
        X = X.toarray()
    # --------------------------------------

    if oversample:
        X, y = RandomOverSampler().fit_resample(X, y)

    return X, y

X_train, y_train = scale_dataset(train, oversample=True)
X_valid, y_valid = scale_dataset(valid, oversample=False)
X_test,  y_test  = scale_dataset(test, oversample=False)

nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_valid)
print(classification_report(y_valid, y_pred))