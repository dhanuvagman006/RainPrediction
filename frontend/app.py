from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "./rainfall_prediction_model.pkl"

with open(MODEL_PATH, 'rb') as file:
    model_data = pickle.load(file)
    model = model_data["model"]
    feature_names = model_data["feature_names"]

@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")  # Serve the frontend

@app.route('/predict', methods=['POST'])  # Separate route for predictions
def predict():
    try:
        data = request.get_json()
        inputs = [float(data.get(feature, 0)) for feature in feature_names]
        input_df = pd.DataFrame([inputs], columns=feature_names)
        prediction = model.predict(input_df)[0]
        prediction_result = "Rainfall" if prediction == 1 else "No Rainfall"
        return jsonify({"prediction": prediction_result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
