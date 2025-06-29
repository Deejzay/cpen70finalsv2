# Taal Lake Water Quality Prediction

This project implements deep learning models to predict water quality in Taal Lake, taking into account various environmental and volcanic factors.

## Project Structure
```
project/
├── data/
│   ├── raw/                    # Raw datasets
│   └── processed/              # Cleaned and normalized data
├── src/
│   ├── data/
│   │   ├── preprocessing.py    # Data cleaning and normalization
│   │   └── wqi_calculator.py   # WQI calculation
│   ├── models/
│   │   ├── cnn_model.py
│   │   ├── lstm_model.py
│   │   └── hybrid_model.py
│   └── visualization/
│       └── plotting.py
├── requirements.txt
└── README.md
```

## Datasets
1. Climate Data (Ambulong-Monthly-Data.csv)
   - Monthly climate parameters from 2013-2023
   - Includes rainfall, temperature, humidity, wind data

2. Volcanic Activity Data
   - SO2 Flux (2020-2024)
   - CO2 Flux (2013-2019)

3. Water Parameters (2013-2025)
   - Various water quality measurements

## Models
1. CNN Model
   - 1D Convolutional layers for feature extraction
   - MaxPooling and Dense layers

2. LSTM Model
   - LSTM layers for sequence learning
   - Dense layers for prediction

3. Hybrid CNN-LSTM Model
   - Combines CNN and LSTM architectures
   - Enhanced feature extraction and sequence learning

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run preprocessing:
```bash
python src/data/preprocessing.py
```

## Usage
[To be added as we implement the models]

## Results
[To be added after model implementation and evaluation] 