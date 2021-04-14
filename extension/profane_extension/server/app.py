from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import numpy as np
from ProfanityChecker import ProfanityChecker


model = ProfanityChecker()
app = Flask(__name__)
CORS(app)

def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route("/")
def index():
	return "hello world"

@app.route("/predict", methods = ['GET', 'POST'])
# @cross_origin()
def predict():
	if request.method == 'POST':
		# data = request
		data = request.form['data']
		# print(request.form['data'])
		res = model.predict_sentiment(data)
		print(data, res)
		return jsonify({"success":"true", "target": res})

app.after_request(add_cors_headers)

if __name__ == "__main__":
	model.init()
	app.run(port = 3000, debug=True)