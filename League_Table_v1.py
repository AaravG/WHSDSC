import os
import tensorflow as tf
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import sqlite3
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import random

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

SEED = 42

# Python
random.seed(SEED)

# NumPy
np.random.seed(SEED)

# TensorFlow
tf.random.set_seed(SEED)

# Optional: Full determinism (slightly slower)inst
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

cat_cols = [
  "home_team","away_team","home_off_line","home_def_pairing",
  "away_off_line","away_def_pairing","home_goalie","away_goalie"
]

num_cols = [
  "went_ot","toi","home_assists","home_shots","home_xg","home_max_xg",
  "away_assists","away_shots","away_xg","away_max_xg",
  "home_penalties_committed","home_penalty_minutes",
  "away_penalties_committed","away_penalty_minutes", "home_goals", "away_goals"
]



conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")
df = pd.read_sql("SELECT * FROM kNN", conn)
conn.close()

df = df.sample(frac=1)
train = df.iloc[:int(0.6*len(df))]
valid = df.iloc[int(0.6*len(df)):int(0.8*len(df))]
test  = df.iloc[int(0.8*len(df)):]

# -------------------------------
# BUILD PREPROCESSOR (ONCE)
# -------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])
X_train_raw = train[cat_cols + num_cols]
y_train = train["home_win"]

X_valid_raw = valid[cat_cols + num_cols]
y_valid = valid["home_win"]

X_test_raw  = test[cat_cols + num_cols]
y_test  = test["home_win"]

# Fit encoder ONLY on training
preprocessor.fit(X_train_raw)

# Transform all splits with SAME encoder
X_train = preprocessor.transform(X_train_raw)
X_valid = preprocessor.transform(X_valid_raw)
X_test  = preprocessor.transform(X_test_raw)

# Convert sparse → dense (Keras needs dense)
X_train = X_train.toarray()
X_valid = X_valid.toarray()
X_test  = X_test.toarray()

ros = RandomOverSampler()
X_train, y_train = ros.fit_resample(X_train, y_train)


def plot_history(history, name):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
  ax1.plot(history.history['loss'], label='loss')
  ax1.plot(history.history['val_loss'], label='val_loss')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Binary crossentropy')
  ax1.grid(True)

  ax2.plot(history.history['accuracy'], label='accuracy')
  ax2.plot(history.history['val_accuracy'], label='val_accuracy')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.grid(True)
  # Save the plot as a PNG file
  plt.savefig(f'{name}.png', dpi=300, bbox_inches='tight')
  
def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs):
  nn_model = tf.keras.Sequential([
      tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(X_train.shape[1],)),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Dense(num_nodes, activation='relu'),
      tf.keras.layers.Dropout(dropout_prob),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                  metrics=['accuracy'])
  history = nn_model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1
  )

  return nn_model, history

least_val_loss = float('inf')
least_loss_model = None
epochs=100
for num_nodes in [32]:
  for dropout_prob in[0.2]:  #0, 0.2
    for lr in [0.001]:
      for batch_size in [32]:
        print(f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
        model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
        #plot_history(history, f"nodes {num_nodes} dropout {dropout_prob} lr {lr} batch size {batch_size}")
        val_loss = model.evaluate(X_valid, y_valid)[0]
        if val_loss < least_val_loss:
          least_val_loss = val_loss
          least_loss_model = model
        y_pred = model.predict(X_valid)
        y_pred = (y_pred > 0.5).astype(int).reshape(-1,)
        print(classification_report(y_valid, y_pred))
        
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Background sample (larger = more accurate)
# -------------------------------------------------
background_size = 100
np.random.seed(SEED)
background_idx = np.random.choice(X_train.shape[0], background_size, replace=False)
background = X_train[background_idx]

# -------------------------------------------------
# 2. KernelExplainer (Most rigorous SHAP method)
# -------------------------------------------------
explainer = shap.KernelExplainer(
    least_loss_model.predict,
    background
)

# -------------------------------------------------
# 3. Select validation sample
# -------------------------------------------------
X_sample = X_valid

shap_values = explainer.shap_values(
    X_sample,
    nsamples=2000  # increase for more accuracy
)

# -------------------------------------------------
# 4. Safe SHAP extraction (handles all formats)
# -------------------------------------------------
if isinstance(shap_values, list):
    shap_array = shap_values[0]
else:
    shap_array = shap_values

shap_array = np.array(shap_array)

# Remove trailing singleton dimension (fix for your error)
if shap_array.ndim == 3 and shap_array.shape[2] == 1:
    shap_array = shap_array.squeeze(-1)

print("Final SHAP shape:", shap_array.shape)

# -------------------------------------------------
# 5. Align with feature names
# -------------------------------------------------
feature_names = preprocessor.get_feature_names_out()

if shap_array.shape[1] != len(feature_names):
    raise ValueError(
        f"Feature mismatch: SHAP={shap_array.shape[1]}, "
        f"Features={len(feature_names)}"
    )

# -------------------------------------------------
# 6. Global importance (mean |SHAP|)
# -------------------------------------------------
mean_abs_shap = np.mean(np.abs(shap_array), axis=0)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": mean_abs_shap
}).sort_values(by="importance", ascending=False)

print("\nAll Features (more important to less important):")
print(importance_df.head(len(X_valid)))

# -------------------------------------------------
# 7. SHAP summary plot
# -------------------------------------------------
# shap.summary_plot(
#     shap_array,
#     X_sample,
#     feature_names=feature_names
# )



# -------------------------------------------------
# ASSUMPTIONS (already computed earlier)
# -------------------------------------------------
# shap_array → shape (samples, 164)
# feature_names → length 164
# X_valid → processed feature matrix (same order as SHAP)

# -------------------------------------------------
# 1. Compute SHAP global weights (ALL features)
# -------------------------------------------------
mean_abs_shap = np.mean(np.abs(shap_array), axis=0)

# Normalize weights (optional but recommended)
weights = mean_abs_shap / np.sum(mean_abs_shap)

# Convert to dictionary
weight_dict = dict(zip(feature_names, weights))

# -------------------------------------------------
# 2. Convert validation matrix to DataFrame
# -------------------------------------------------
X_valid_df = pd.DataFrame(X_valid, columns=feature_names)

# -------------------------------------------------
# 3. Compute SHAP-weighted score per row
# -------------------------------------------------
def compute_weighted_score(row):
    return np.sum(row.values * weights)

X_valid_df["weighted_score"] = X_valid_df.apply(compute_weighted_score, axis=1)

# -------------------------------------------------
# 4. Identify team columns (one-hot encoded)
# -------------------------------------------------
home_team_cols = [col for col in feature_names if "cat__home_team_" in col]
away_team_cols = [col for col in feature_names if "cat__away_team_" in col]

# Extract team names from column names
def extract_team_name(col):
    return col.split("cat__home_team_")[-1] if "home_team" in col \
           else col.split("cat__away_team_")[-1]

# -------------------------------------------------
# 5. Aggregate team scores
# -------------------------------------------------
team_scores = {}

for idx, row in X_valid_df.iterrows():
    score = row["weighted_score"]
    
    # Identify home team
    for col in home_team_cols:
        if row[col] == 1:
            team = extract_team_name(col)
            team_scores.setdefault(team, []).append(score)
    
    # Identify away team
    for col in away_team_cols:
        if row[col] == 1:
            team = extract_team_name(col)
            team_scores.setdefault(team, []).append(score)

# Average score per team
final_team_scores = {
    team: np.mean(scores)
    for team, scores in team_scores.items()
}

team_rankings = pd.Series(final_team_scores).sort_values(ascending=False)

print("\n===== TEAM RANKINGS (SHAP-Weighted) =====")
print(team_rankings)
