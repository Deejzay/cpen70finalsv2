import joblib

scaler = joblib.load('data/processed/scaler.pkl')
print('Scaler feature names:', getattr(scaler, 'feature_names_in_', 'No feature_names_in_ attribute')) 