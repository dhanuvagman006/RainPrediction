from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "./rainfall_prediction_model.pkl"

with open(MODEL_PATH, 'rb') as file:
    model_data = pickle.load(file)
    model = model_data["model"]
    feature_names = model_data["feature_names"]

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_result = None
    if request.method == 'POST':
        try:
            inputs = [float(request.form.get(feature, 0)) for feature in feature_names]
            input_df = pd.DataFrame([inputs], columns=feature_names)
            prediction = model.predict(input_df)[0]
            prediction_result = "Rainfall" if prediction == 1 else "No Rainfall"
        except Exception as e:
            prediction_result = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=prediction_result)

if __name__ == '__main__':
    app.run(debug=True)
