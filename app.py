import pickle, json
import numpy as np
from flask import Flask,render_template,request


app = Flask(__name__)

@app.route('/')
def home():
    models = ['random_forest', 'decision_tree', 'naive_bayes']
    return render_template('index.html', models = models)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = []
    params =  request.form.values()
    model_name = next(params)
    model = pickle.load(open('./log/' + model_name + '.sav', 'rb'))
    for x in params:
        int_features.append(float(x))
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    prediction_output = True if prediction is [1] else False 
    prediction_probability = round(model.predict_proba(final_features)[0][1] * 100, 2)


    return render_template('index.html', prediction_text=prediction_output, prediction_probability= prediction_probability)

