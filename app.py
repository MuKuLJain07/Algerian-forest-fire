import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)


# import our models
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get('temperature'))
        RH = float(request.form.get('rh'))
        Ws = float(request.form.get('ws'))
        Rain = float(request.form.get('rain'))
        FFMC = float(request.form.get('ffmc'))
        DMC = float(request.form.get('dmc'))
        ISI = float(request.form.get('isi'))
        Classes = float(request.form.get('classes'))
        Region = float(request.form.get('region'))


        # standardize Data
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('predict.html',results = result[0])
    else:
        return render_template('predict.html')


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")