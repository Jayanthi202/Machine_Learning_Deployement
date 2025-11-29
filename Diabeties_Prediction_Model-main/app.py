import numpy as np
import joblib
from flask import Flask, request, render_template

# Load your trained model (saved using joblib)
model = joblib.load('diabetes.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get all input values
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        # 2. Predict directly (NO SCALER REQUIRED)
        prediction = model.predict(final_features)[0]

        # 3. Output text
        if prediction == 1:
            result = "The model predicts the patient is likely to have Diabetes."
        else:
            result = "The model predicts the patient is NOT likely to have Diabetes."

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
