import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
import joblib

# Load Netflix churn dataset
df = pd.read_csv("netflix_churn.csv")
df.dropna(inplace=True)

# Features and label
X = df[["monthly_hours", "data_usage", "plan_price"]]
y = df["churn"]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, "netflix_scaler.pkl")

# Custom callback: stop if accuracy decreases
class StopIfAccuracyDecreases(Callback):
    def on_train_begin(self, logs=None):
        self.prev_acc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy')
        if acc is not None:
            if acc < self.prev_acc:
                print(f"\n⛔ Stopping early at epoch {epoch+1} — accuracy dropped from {self.prev_acc:.4f} to {acc:.4f}")
                self.model.stop_training = True
            else:
                self.prev_acc = acc

# Build model
model = Sequential()
model.add(Dense(32, input_dim=X.shape[1], activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model with early stopping
model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, verbose=1, callbacks=[StopIfAccuracyDecreases()])

# Evaluate and print
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\n✅ Final Test Accuracy: {round(accuracy * 100, 2)}%")

# Save model
model.save("netflix_churn_model.h5")
print("✔️ Model and scaler saved successfully!")
model.summary()