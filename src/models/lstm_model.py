import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print('Loading data...')
X_train = np.load('data/processed/X_train.npy', allow_pickle=True)
X_test = np.load('data/processed/X_test.npy', allow_pickle=True)
y_train = np.load('data/processed/y_train.npy', allow_pickle=True)
y_test = np.load('data/processed/y_test.npy', allow_pickle=True)

print('Converting data to float32...')
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

print('Building LSTM model...')
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(1)
])

print('Compiling model...')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

print('Training model...')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=2
)

print('Predicting on test set...')
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('\nTest MSE:', mse, flush=True)
print('Test RMSE:', rmse, flush=True)
print('Test MAE:', mae, flush=True)
print('Test R²:', r2, flush=True)

print('Saving model...')
model.save('models/lstm_model.h5')
print('Done.') 