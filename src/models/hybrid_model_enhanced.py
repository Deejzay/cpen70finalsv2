import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the enhanced data
X_train = np.load('data/processed/X_train_enhanced.npy', allow_pickle=True)
X_test = np.load('data/processed/X_test_enhanced.npy', allow_pickle=True)
y_train_wqi = np.load('data/processed/y_train_wqi.npy', allow_pickle=True)
y_test_wqi = np.load('data/processed/y_test_wqi.npy', allow_pickle=True)
y_train_pollutant = np.load('data/processed/y_train_pollutant_level.npy', allow_pickle=True)
y_test_pollutant = np.load('data/processed/y_test_pollutant_level.npy', allow_pickle=True)

# Ensure data is float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train_wqi = y_train_wqi.astype(np.float32)
y_test_wqi = y_test_wqi.astype(np.float32)
y_train_pollutant = y_train_pollutant.astype(np.float32)
y_test_pollutant = y_test_pollutant.astype(np.float32)

print(f"Training data shape: {X_train.shape}")
print(f"WQI training targets: {y_train_wqi.shape}")
print(f"Pollutant training targets: {y_train_pollutant.shape}")

# Define the enhanced hybrid model with multi-output
def create_enhanced_hybrid_model(input_shape, num_targets=2):
    """
    Create a hybrid CNN-LSTM model with multi-output for WQI and Pollutant Level prediction
    """
    inputs = layers.Input(shape=input_shape)
    
    # CNN branch for feature extraction
    cnn = layers.Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    cnn = layers.MaxPooling1D(pool_size=2)(cnn)
    cnn = layers.Dropout(0.2)(cnn)
    cnn = layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(cnn)
    cnn = layers.GlobalAveragePooling1D()(cnn)
    
    # LSTM branch for temporal patterns
    lstm = layers.LSTM(64, activation='tanh', return_sequences=True)(inputs)
    lstm = layers.Dropout(0.2)(lstm)
    lstm = layers.LSTM(32, activation='tanh', return_sequences=True)(lstm)
    
    # Attention mechanism
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(lstm, lstm)
    attention = layers.GlobalAveragePooling1D()(attention)
    
    # Combine CNN and LSTM features
    combined = layers.Concatenate()([cnn, attention])
    
    # Shared dense layers
    x = layers.Dense(256, activation='relu')(combined)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    
    # Separate output layers for each target
    wqi_output = layers.Dense(64, activation='relu', name='wqi_dense')(x)
    wqi_output = layers.Dense(1, name='wqi_output')(wqi_output)
    
    pollutant_output = layers.Dense(64, activation='relu', name='pollutant_dense')(x)
    pollutant_output = layers.Dense(1, name='pollutant_output')(pollutant_output)
    
    # Create model with multiple outputs
    model = keras.Model(inputs=inputs, outputs=[wqi_output, pollutant_output])
    
    return model

# Create the enhanced model
model = create_enhanced_hybrid_model((X_train.shape[1], X_train.shape[2]))

# Compile with custom loss weights
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={
        'wqi_output': 'mse',
        'pollutant_output': 'mse'
    },
    loss_weights={
        'wqi_output': 1.0,
        'pollutant_output': 1.0
    },
    metrics={
        'wqi_output': ['mae'],
        'pollutant_output': ['mae']
    }
)

# Print model summary
print("\nEnhanced Hybrid Model Architecture:")
model.summary()

# Prepare training data
training_data = {
    'wqi_output': y_train_wqi,
    'pollutant_output': y_train_pollutant
}

validation_data = {
    'wqi_output': y_test_wqi,
    'pollutant_output': y_test_pollutant
}

# Train the model
print("\nTraining enhanced hybrid model...")
history = model.fit(
    X_train, training_data,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# Evaluate the model
print("\nEvaluating enhanced hybrid model...")
test_predictions = model.predict(X_test)
wqi_pred, pollutant_pred = test_predictions

# Calculate metrics for WQI
wqi_mse = mean_squared_error(y_test_wqi, wqi_pred)
wqi_rmse = np.sqrt(wqi_mse)
wqi_mae = mean_absolute_error(y_test_wqi, wqi_pred)
wqi_r2 = r2_score(y_test_wqi, wqi_pred)

# Calculate metrics for Pollutant Level
pollutant_mse = mean_squared_error(y_test_pollutant, pollutant_pred)
pollutant_rmse = np.sqrt(pollutant_mse)
pollutant_mae = mean_absolute_error(y_test_pollutant, pollutant_pred)
pollutant_r2 = r2_score(y_test_pollutant, pollutant_pred)

# Print results
print("\n" + "="*60)
print("ENHANCED HYBRID MODEL PERFORMANCE")
print("="*60)

print("\nWQI Prediction Performance:")
print(f"MSE: {wqi_mse:.6f}")
print(f"RMSE: {wqi_rmse:.6f}")
print(f"MAE: {wqi_mae:.6f}")
print(f"R²: {wqi_r2:.6f}")

print("\nPollutant Level Prediction Performance:")
print(f"MSE: {pollutant_mse:.6f}")
print(f"RMSE: {pollutant_rmse:.6f}")
print(f"MAE: {pollutant_mae:.6f}")
print(f"R²: {pollutant_r2:.6f}")

# Calculate overall performance
overall_mse = (wqi_mse + pollutant_mse) / 2
overall_rmse = (wqi_rmse + pollutant_rmse) / 2
overall_mae = (wqi_mae + pollutant_mae) / 2
overall_r2 = (wqi_r2 + pollutant_r2) / 2

print("\nOverall Model Performance (Average):")
print(f"Average MSE: {overall_mse:.6f}")
print(f"Average RMSE: {overall_rmse:.6f}")
print(f"Average MAE: {overall_mae:.6f}")
print(f"Average R²: {overall_r2:.6f}")

# Save the enhanced model
model.save('models/hybrid_model_enhanced_multi_output.h5')
print("\nEnhanced hybrid model saved as 'models/hybrid_model_enhanced_multi_output.h5'")

# Save prediction results for analysis
results = {
    'wqi_actual': y_test_wqi,
    'wqi_predicted': wqi_pred.flatten(),
    'pollutant_actual': y_test_pollutant,
    'pollutant_predicted': pollutant_pred.flatten()
}

np.save('data/processed/enhanced_model_predictions.npy', results)
print("Prediction results saved for analysis")

# Print sample predictions
print("\nSample Predictions (First 5 test samples):")
print("Index | WQI (Actual) | WQI (Pred) | Pollutant (Actual) | Pollutant (Pred)")
print("-" * 70)
for i in range(min(5, len(y_test_wqi))):
    print(f"{i:5d} | {y_test_wqi[i]:11.3f} | {wqi_pred[i][0]:10.3f} | {y_test_pollutant[i]:17.3f} | {pollutant_pred[i][0]:16.3f}")

print("\nEnhanced hybrid model training completed successfully!") 