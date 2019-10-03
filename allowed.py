import os
import io
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import (Xception, preprocess_input, decode_predictions)
from tensorflow.keras import backend as K

from flask import Flask, request, redirect, url_for, jsonify, render_template

from werkzeug.utils import secure_filename

# use os path join
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None
graph = None

#FILE VALIDATION
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    global model
    global graph
    model = Xception(weights="imagenet")
    graph = K.get_session().graph

load_model()


def prepare_image(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # return the processed image
    return img


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    display = ""
    imagefile = ""
    data = {"success": False}
    if request.method == 'POST':
        if 'file' not in request.files:
            # flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # create a path to the uploads folder
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(filepath)

            # Load the saved image using Keras and resize it to the Xception
            # format of 299x299 pixels
            global image
            image_size = (299, 299)
            im = image.load_img(filepath, target_size=image_size, grayscale=False)

            # preprocess the image and prepare it for classification
            image = prepare_image(im)
            # imagefile = "Uplaods/"+filename.replace('\\', '/')
            imagefile = filename
          

            global graph
            with graph.as_default():
                preds = model.predict(image)
                results = decode_predictions(preds)
                data["predictions"] = []

                # loop over the results and add them to the list of
                # returned predictions
                for (imagenetID, label, prob) in results[0]:
                    r = {"label": label, "probability": float(prob)}
                    data["predictions"].append(r)

                # indicate that the request was a success
                data["success"] = True

       
        
        display = data["predictions"][0]["label"]

    return render_template("index.html",display=display,imagefile=imagefile)



if __name__ == "__main__":
    app.run(debug=True)
