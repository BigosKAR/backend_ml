import os
import pickle
import numpy as np
import joblib

# Load the model once at module level (not every time function is called)
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'pc_price_predictor_selected.pkl')
features_path = os.path.join(base_dir, 'selected_feature_columns.pkl')

model = joblib.load(model_path)
feature_columns = joblib.load(features_path)

def pricePrediction(prediction_features):
    """
    prediction_features: list of feature values
    """
    # Convert to 2D array if it's a flat list
    input_array = np.array(prediction_features).reshape(1, -1)

    prediction = model.predict(input_array)
    print(f"The prediction is: {prediction}")

    return round(float(prediction[0]), 2)
