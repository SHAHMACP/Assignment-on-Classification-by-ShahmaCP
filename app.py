import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('knn_model.pkl','rb'))


@app.route('/')
def home():
    return render_template("Iris.html")


@app.route("/predict", methods=['POST', 'GET'])
def predict():
    sep_len = float(request.form['sl'])
    sep_wid = float(request.form['sw'])
    pet_len = float(request.form['pl'])
    pet_wid = float(request.form['pw'])

    result = model.predict([[sep_len, sep_wid, pet_len, pet_wid]])
    output = result[0]

    return render_template("Iris.html", prediction_text="The Iris Species is: {}".format(output))


if __name__ == "__main__":
    app.run()

