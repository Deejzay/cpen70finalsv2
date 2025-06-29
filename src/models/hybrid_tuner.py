import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

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

def build_hybrid_model(hp):
    # Input layer
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
    
    # CNN branch
    cnn = layers.Conv1D(
        filters=hp.Int('cnn_filters', min_value=32, max_value=128, step=32),
        kernel_size=hp.Int('kernel_size', min_value=2, max_value=5),
        activation='relu',
        padding='same'
    )(inputs)
    cnn = layers.MaxPooling1D(pool_size=2)(cnn)
    cnn = layers.Dropout(hp.Float('cnn_dropout', min_value=0.1, max_value=0.5, step=0.1))(cnn)
    
    # LSTM branch
    lstm = layers.LSTM(
        units=hp.Int('lstm_units', min_value=32, max_value=128, step=32),
        return_sequences=True
    )(inputs)
    lstm = layers.Dropout(hp.Float('lstm_dropout', min_value=0.1, max_value=0.5, step=0.1))(lstm)
    
    # Attention mechanism
    attention = layers.MultiHeadAttention(
        num_heads=hp.Int('num_heads', min_value=2, max_value=8, step=2),
        key_dim=hp.Int('key_dim', min_value=16, max_value=64, step=16)
    )(lstm, lstm)
    
    # Reshape CNN output to match attention output
    cnn_reshaped = layers.Reshape((X_train.shape[1], -1))(cnn)
    
    # Combine branches
    combined = layers.Concatenate()([cnn_reshaped, attention])
    
    # Dense layers
    x = layers.Dense(
        units=hp.Int('dense_units', min_value=32, max_value=128, step=32),
        activation='relu'
    )(combined)
    x = layers.Dropout(hp.Float('dense_dropout', min_value=0.1, max_value=0.5, step=0.1))(x)
    
    # Output layer
    outputs = layers.Dense(1)(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        ),
        loss='mse'
    )
    
    return model

# Add batch_size as a hyperparameter to the tuner
search_batch_size = [16, 32, 64, 128]

# Define the tuner
tuner = kt.Hyperband(
    build_hybrid_model,
    objective='val_loss',
    max_epochs=100,
    factor=3,
    directory='hybrid_tuning',
    project_name='hybrid_model'
)

# Define early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Search for best hyperparameters
# We'll loop over batch_size manually since it's not part of the model hyperparameters
for batch_size in search_batch_size:
    print(f"\nSearching with batch size: {batch_size}")
    tuner.search(
        X_train, y_train,
        epochs=100,
        validation_split=0.2,
        callbacks=[early_stopping],
        batch_size=batch_size
    )

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\nBest hyperparameters:")
for param, value in best_hps.values.items():
    print(f"{param}: {value}")

# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Find the best batch size by checking the tuner trials
best_trial = tuner.oracle.get_best_trials(1)[0]
best_batch_size = best_trial.hyperparameters.get('batch_size', 32)  # Default to 32 if not found
print(f"Best batch size: {best_batch_size}")

# Train the model with the best hyperparameters
history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_split=0.2,
    callbacks=[early_stopping],
    batch_size=best_batch_size
)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation metrics with best hyperparameters:")
print(f"Test MSE: {mse}")
print(f"Test RMSE: {rmse}")
print(f"Test MAE: {mae}")
print(f"Test R²: {r2}")

# Save the best model
model.save('models/hybrid_model_tuned.h5') 