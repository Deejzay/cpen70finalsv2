import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data
X_train = np.load('data/processed/X_train.npy', allow_pickle=True)
X_test = np.load('data/processed/X_test.npy', allow_pickle=True)
y_train = np.load('data/processed/y_train.npy', allow_pickle=True)
y_test = np.load('data/processed/y_test.npy', allow_pickle=True)

# Ensure data is float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

# Define the hybrid model
inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
cnn = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
cnn = layers.MaxPooling1D(pool_size=2)(cnn)
cnn = layers.GlobalAveragePooling1D()(cnn)

lstm = layers.LSTM(64, activation='tanh', return_sequences=True)(inputs)
attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(lstm, lstm)
attention = layers.GlobalAveragePooling1D()(attention)

combined = layers.Concatenate()([cnn, attention])
x = layers.Dense(128, activation='relu')(combined)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Train the model for the full number of epochs (no early stopping)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTest MSE:", mse)
print("Test RMSE:", rmse)
print("Test MAE:", mae)
print("Test R²:", r2)

model.save('models/hybrid_model_no_early_stopping.h5') 