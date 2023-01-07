import warnings
warnings.filterwarnings("ignore")

# imports
from flask import Flask, render_template, request, url_for, jsonify, redirect
from model import *

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict_image_file():
    try:
        if request.method == 'POST':
            img = preprocess_img(request)
            pred = predict_result(img)
            return pred
    except:
        error = "File cannot be processed."
        return error
    
    
@app.route('/classify_url', methods=['GET'])
def predict_image_url():
    pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
