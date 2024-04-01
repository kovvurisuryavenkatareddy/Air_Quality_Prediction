from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    soi = float(request.form['soi'])
    noi = float(request.form['noi'])
    SPMi = float(request.form['SPMi'])

    # Make a prediction
    prediction = model.predict([[soi, noi, SPMi]])
    prediction = np.array(prediction, dtype=int)  
    predicted_class = label_encoder.inverse_transform(prediction)[0]

    return render_template('result.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)