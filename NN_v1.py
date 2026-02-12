import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(8)  #config for faster training on Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz (1.99 GHz), adjust as needed
tf.config.threading.set_inter_op_parallelism_threads(2)  #config for faster training on CPU, adjust as needed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import sqlite3
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


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
  plt.close('all')  # Close the plot to free up memory
  
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
    X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0
  )

  return nn_model, history

least_val_loss = float('inf')
least_loss_model = None
epochs=100
for num_nodes in [128]:
  for dropout_prob in[0.1, 0.2, 0.3]:  #0, 0.2
    for lr in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
      for batch_size in [16, 32, 64, 128]:
        print(f"{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
        model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
        plot_history(history, f"nodes {num_nodes} dropout {dropout_prob} lr {lr} batch size {batch_size}")
        val_loss = model.evaluate(X_valid, y_valid)[0]
        if val_loss < least_val_loss:
          least_val_loss = val_loss
          least_loss_model = model
        y_pred = model.predict(X_valid)
        y_pred = (y_pred > 0.5).astype(int).reshape(-1,)
        with open("accuracy_list_NN.txt", "a") as file:
            file.write(f"""\n{num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}
                       {classification_report(y_valid, y_pred)}
                       {model.evaluate(X_valid, y_valid)}""")

y_pred = least_loss_model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1,)
print(classification_report(y_test, y_pred))