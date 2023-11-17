# NFL Prediction Models

## Overview
This repository contains two key files used for predictive modeling in the context of NFL games: `trained_nfl_model.pkl` and `data_scaler.pkl`. These files are essential components of a machine learning pipeline designed to predict outcomes or other relevant metrics in NFL games.

## File Descriptions

### 1. trained_nfl_model.pkl
This file is a serialized version of the trained Ridge regression model. It is used for making predictions related to NFL games. The model has been trained on historical NFL game data, factoring in various features such as team statistics, player performance, and other relevant metrics.

#### Usage:
To use this model for prediction, follow these steps:
```python
import joblib

# Load the model
model = joblib.load('trained_nfl_model.pkl')

# Make predictions
# Ensure that X_test is preprocessed and has the same format as the training data
predictions = model.predict(X_test)
```

2. data_scaler.pkl
This file is a serialized version of a scaler object (e.g., StandardScaler from scikit-learn) used to normalize or standardize the features used in the Ridge regression model. It's crucial to apply this scaler to any new data before making predictions with trained_nfl_model.pkl.

Usage:

To use the scaler, follow these steps:

```python
import joblib

# Load the scaler
scaler = joblib.load('data_scaler.pkl')

# Scale your data
# X_new is the new data you want to scale
X_scaled = scaler.transform(X_new)
```

#### Requirements

Python 3.x
scikit-learn
joblib