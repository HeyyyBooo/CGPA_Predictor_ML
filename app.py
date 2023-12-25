# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 18:31:06 2023

@author: narze Nishant
"""



import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load("cgpa_predictor.pkl")

df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global df
    
    input_features = [x for x in request.form.values()]
    if(input_features[2]=='single'):
        input_features[2]=0
    else:
        input_features[2]=1
    
    for i in [0,1,2] :
        input_features[i]=int(input_features[i])
    features_value = np.array(input_features)
    #validate input hours
    
    if input_features[0] <0 or input_features[0] >24:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24')
    if input_features[1] <0:
        return render_template('index.html', prediction_text='Atleast Start a day before')   

    output_arr = model.predict([features_value])
    output=output_arr[0].round(2)
    # input and predicted value store in df then save in csv file
    #df= pd.concat([df,pd.DataFrame({'Study Hours':input_features,'Predicted Output':[output]})],ignore_index=True)
    #print(df)   
    #df.to_csv('smp_data_from_app.csv')
    if(input_features[2]==0):
        return render_template('index.html', prediction_text='You will get {} CGPA, when you do study {} hours per day and start {} day before exams. All The Best Hope You get a good CG and Life Partner '.format(output, int(features_value[0]),int(features_value[1])))
    else:
        return render_template('index.html', prediction_text='You will get {} CGPA, when you do study {} hours per day and start {} day before exams. All The Best'.format(output, int(features_value[0]),int(features_value[1])))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
    