from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# -----------------------------
# Load all saved model objects
# -----------------------------
imputer = joblib.load("imputer.joblib")
scaler = joblib.load("scaler.joblib")
best_model = joblib.load("logistic_regression_model.joblib")  # <-- Make sure you exported this correctly

# Columns where zero is invalid
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Feature order used during training
feature_names = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

# Initialize Flask app
app = Flask(__name__)


@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API! Use POST /predict with feature values."


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        if 'features' not in data:
            return jsonify({'error': "Missing 'features' key in JSON"}), 400

        # Convert to correct shape
        input_features = np.array(data['features']).reshape(1, -1)

        # Create DataFrame with correct columns
        input_df = pd.DataFrame(input_features, columns=feature_names)

        # Replace biological zero values with NaN
        for col in zero_columns:
            input_df[col] = input_df[col].replace(0, np.nan)

        # Impute missing values
        input_df[zero_columns] = imputer.transform(input_df[zero_columns])

        # Scale the entire feature set
        scaled_features = scaler.transform(input_df)

        # Make prediction
        prediction = best_model.predict(scaled_features)[0]
        prediction_proba = best_model.predict_proba(scaled_features)[0]

        result = "Diabetes" if prediction == 1 else "No Diabetes"

        return jsonify({
            "prediction": result,
            "probability_no_diabetes": float(prediction_proba[0]),
            "probability_diabetes": float(prediction_proba[1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # REQUIRED for Render
    app.run(host="0.0.0.0", port=5000)
