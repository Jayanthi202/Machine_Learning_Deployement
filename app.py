from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import joblib
import numpy as np
# Initialize Flask app
app = Flask(__name__)
# Run with ngrok to expose the server
run_with_ngrok(app)
import joblib

# Save the imputer object
joblib.dump(imputer, 'imputer.joblib')
print("Imputer successfully exported as 'imputer.joblib'")

# Save the scaler object
joblib.dump(scaler, 'scaler.joblib')
print("Scaler successfully exported as 'scaler.joblib'")
# Define the columns where zero values are biologically implausible (consistent with preprocessing)
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Initialize Flask app
app = Flask(__name__)
# Run with ngrok to expose the server
run_with_ngrok(app)

@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API! Use /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Assuming the input data is a list of feature values in the correct order
        input_features = np.array(data['features']).reshape(1, -1)

        # Create a DataFrame for consistent preprocessing, using the original column names
        # Need to ensure the order of features is consistent with training
        # The `X.columns` from the kernel state holds the correct order.
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        input_df = pd.DataFrame(input_features, columns=feature_names)

        # Apply imputer for zero_columns if any are present in the input_df
        for col in zero_columns:
            if col in input_df.columns:
                input_df[col] = input_df[col].replace(0, np.nan)
        
        # Impute missing values (NaNs from previous step or actual NaNs) using the loaded imputer
        input_df[zero_columns] = imputer.transform(input_df[zero_columns])

        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(input_df)

        # Make prediction
        prediction = best_model.predict(scaled_features)
        prediction_proba = best_model.predict_proba(scaled_features)

        outcome = 'Diabetes' if prediction[0] == 1 else 'No Diabetes'

        return jsonify({
            'prediction': outcome,
            'probability_no_diabetes': prediction_proba[0][0],
            'probability_diabetes': prediction_proba[0][1]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# This line is not needed in Colab when using run_with_ngrok
if __name__ == '__main__':
  app.run(debug=True)
