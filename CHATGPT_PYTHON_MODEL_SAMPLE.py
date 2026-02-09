# =========================
# 1. Imports
# =========================
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import StandardScaler

# =========================
# 2. Load data from SQL
# =========================

engine = create_engine("sqlite:///E:/Aarav PC Data/Code/WHSDSC/whl_2025.db")  # change if needed

data = pd.read_sql("SELECT * FROM data", engine)
games = pd.read_sql("SELECT * FROM game_summary", engine)

# =========================
# 3. Merge tables
# =========================
df = data.merge(
    games[["game_id", "home_win"]],
    on="game_id",
    how="inner"
)

# =========================
# 4. Feature engineering
# =========================
df["xg_diff"] = df["home_xg"] - df["away_xg"]
df["shots_diff"] = df["home_shots"] - df["away_shots"]
df["goals_diff"] = df["home_goals"] - df["away_goals"]
df["penalty_minutes_diff"] = (
    df["home_penalty_minutes"] - df["away_penalty_minutes"]
)

# =========================
# 5. Select features
# =========================
numeric_features = [
    "xg_diff",
    "shots_diff",
    "penalty_minutes_diff",
    "toi",
    "went_ot"
]

categorical_features = [
    "home_team",
    "away_team",
    "home_goalie",
    "away_goalie"
]

X = df[numeric_features + categorical_features]
y = df["home_win"]

# =========================
# 6. Preprocessing
# =========================
numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# =========================
# 7. Model
# =========================
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

# =========================
# 8. Train / test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 9. Train
# =========================
pipeline.fit(X_train, y_train)

# =========================
# 10. Evaluate
# =========================
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_prob))
print(classification_report(y_test, y_pred))

# =========================
# 11. Single-game prediction
# =========================
sample_game = X_test.iloc[[0]]
prob_home_win = pipeline.predict_proba(sample_game)[0][1]
print("Predicted home win probability:", round(prob_home_win, 3))
