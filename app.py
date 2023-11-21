from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Year=int(request.form.get('Year')),
            Kilometers_Driven=int(request.form.get('Kilometers_Driven')),
            Mileage=float(request.form.get('Mileage')),
            Engine=int(request.form.get('Engine')),
            Power=int(request.form.get('Power')),
            Seats=int(request.form.get('Seats')),
            Location=request.form.get('Location'),
            Fuel_Type=request.form.get('Fuel_Type'),
            Transmission=request.form.get('Transmission'),
            Brand=request.form.get('Brand'),
            Owner_Type=request.form.get('Owner_Type')   
        )
        
        pred_df = data.get_data_as_df()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html',results = results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)