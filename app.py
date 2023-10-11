from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import os


from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application
app.debug = False



@app.route('/')
def index():
    print("In index route")   # Debugging print statement
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    print("In predictdata route")   # Debugging print statement
    if request.method == 'GET':
        print("GET request")  # Debugging print statement
        return render_template('home.html')
    else:
        print("POST request")  # Debugging print statement
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race/ethnicity'),
            parental_level_of_education=request.form.get(
                'parental level of education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get(
                'test preparation course'),
            reading_score=float(request.form.get('writing score')),
            writing_score=float(request.form.get('reading score'))

        )
        pred_df = data.get_data_as_data_frame()
        print(f"Predict DataFrame: {pred_df}")   # Debugging print statement

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")  # Debugging print statement
        results = predict_pipeline.predict(pred_df)
        # Debugging print statement
        print(f"After Prediction, results: {results}")
        return render_template('home.html', results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0")
