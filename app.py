from flask import Flask, render_template, request, url_for, jsonify, redirect
import numpy as np
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import base64
import io
import os
import sys
from werkzeug.utils import secure_filename
import imp
import train
import logging

################################# FROM JUPYTER NOTEBOOK #########################################
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import h5py
import sklearn
from sklearn.model_selection import train_test_split
from collections import Counter
import glob
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import train
from random import shuffle
import imp
import os
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

################################# ^^^ FROM JUPYTER NOTEBOOK ^^^ #########################################


#############################################################################################################
#                     KEEP THE BELOW
#############################################################################################################
from PIL import Image
import base64
import io
import numpy as np
# import torch

def readb64(test_image_base64_encoded):

    base64_decoded = base64.b64decode(test_image_base64_encoded)
    image = Image.open(io.BytesIO(base64_decoded))
    # image_np = np.array(image)
    # image_torch = torch.tensor(np.array(image))
    
    # return image_np
    return image

app = Flask(__name__)


characters = [k.split('/')[2] for k in glob.glob('./characters/*') if len([p for p in glob.glob(k+'/*') if 'edited' in p or 'pic_vid' in p]) > 290]
map_characters = dict(enumerate(characters))
map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel', 
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson', 
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak', 
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}


# MODEL_PATH = './models/weights.best.hdf5'
# imp.reload(train)
# model = train.load_model_from_checkpoint('./models/weights.best.hdf5', six_conv=True)

app.logger.info("checkpoint_2_PRE - Model Loading COMPLETE!")

def file_predict(image_path, all_perc=False):
    image = cv2.imread(image_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    pic = cv2.resize(image, (64,64))
    a = model.predict_proba(pic.reshape(1, 64, 64,3))[0]
    if all_perc:
        print('\n'.join(['{} : {}%'.format(map_characters[i], round(k*100)) for i,k in sorted(enumerate(a), key=lambda x:x[1], reverse=True)]))
    else:
        return map_characters[np.argmax(a)].replace('_',' ').title()

import urllib
def url_to_image(url):
    # resp = urllib.request.urlopen(url)
    # image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = np.asarray(url, dtype="uint8")
    # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # image = cv2.imdecode(url, cv2.IMREAD_COLOR)
    return image

def url_predict(url, all_perc=False):
    image = url_to_image(url)
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     plt.imshow(img)
#     plt.show()

    pic = cv2.resize(image, (64,64))
    a = model.predict_proba(pic.reshape(1, 64, 64,3))[0]
    if all_perc:
        print('\n'.join(['{} : {}%'.format(map_characters[i], round(k*100)) for i,k in sorted(enumerate(a), key=lambda x:x[1], reverse=True)]))
    else:
        return map_characters[np.argmax(a)].replace('_',' ').title()
#############################################################################################################
#                     KEEP THE ABOVE
#############################################################################################################

# def file_predict(image_path, all_perc=False):
#     image = cv2.imread(image_path)
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     plt.imshow(img)
#     plt.show()
#     pic = cv2.resize(image, (64,64))
#     a = model.predict_proba(pic.reshape(1, 64, 64,3))[0]
#     if all_perc:
#         print('\n'.join(['{} : {}%'.format(map_characters[i], round(k*100)) for i,k in sorted(enumerate(a), key=lambda x:x[1], reverse=True)]))
#     else:
#         return map_characters[np.argmax(a)].replace('_',' ').title()

# from keras.models import load_model
# import matplotlib.pyplot as plt


# print('About to load the model')
# MODEL_PATH = './models/keras_mnist.h5'

# MODEL_PATH = os.join('models', 'keras_mnist')'models/keras_mnist.h5'
# model = load_model(MODEL_PATH)

## Loading from callbacks
# model = train.load_model_from_checkpoint('./models/weights.best_6conv2.hdf5', six_conv=True)

# print('Model loaded')

def prepare_image(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # return the processed image
    return img


def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    # basepath = os.path.dirname(__file__)
    # file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    file_path = 'models/'+secure_filename(f.filename)
    print('Attempting to save the file at path: ', file_path)
    f.save(file_path)
    print('Saved image at path: ', file_path)
    return file_path


@app.route("/")
def index():
    return render_template("index.html")


# @app.route('/predict', methods=['POST'])
# @app.route("/", methods=['POST'])
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
###################################################################

        # if 'file' not in request.files:
        #     # flash('No file part')
        #     return redirect(request.url)
       
        # Get data from frontend
        jsonData = request.json

        # app.logger.info(jsonData['image'])
        app.logger.info("checkpoint_1 - success")

        # Decode base64 image using readb64 function
        decodedImage = readb64(jsonData['image'])
        
        image_np_1 = np.array(decodedImage)
        image_1 = url_to_image(decodedImage)

        # img = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
        # MODEL_PATH = './models/weights.best.hdf5'
        imp.reload(train)
        model = train.load_model_from_checkpoint('./models/weights.best.hdf5', six_conv=True)

        pic = cv2.resize(image_1, (64,64))
        a = model.predict_proba(pic.reshape(1, 64, 64,3))[0]
        # if all_perc:
        # print('\n'.join(['{} : {}%'.format(map_characters[i], round(k*100)) for i,k in sorted(enumerate(a), key=lambda x:x[1], reverse=True)]))
        # else:
        return map_characters[np.argmax(a)].replace('_',' ').title()

        # img = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
        # image_1 = np.asarray(decodedImage, dtype="uint8")
        # kevin = url_predict(decodedImage)
        return str(image_1==image_np_1)
        # return str(type(image_1))

        # Make a prediction by calling url_predict with decodedImage as a parameter
        # prediction = url_predict(decodedImage)

        image = url_to_image(decodedImage)
        app.logger.info(type(image))
        # kevin = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        # file = request.files['file']
###################################################################
#####  Kevin Experimenting       ##################################
#######################################################################
        # # filename = secure_filename(file.filename)
        # filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # file.save(filepath)

        # # Load the saved image using Keras and resize it to the Xception
        # # format of 299x299 pixels
        # global image
        # image_size = (299, 299)
        # im = image.load_img(filepath, target_size=image_size, grayscale=False)

        # # preprocess the image and prepare it for classification
        # image = prepare_image(im)
        # # imagefile = "Uplaods/"+filename.replace('\\', '/')
        # imagefile = filename
        
        # global graph
        # with graph.as_default():
        #     preds = model.predict(image)
        #     results = decode_predictions(preds)
        #     data["predictions"] = []

        #     # loop over the results and add them to the list of
        #     # returned predictions
        #     for (imagenetID, label, prob) in results[0]:
        #         r = {"label": label, "probability": float(prob)}
        #         data["predictions"].append(r)

        #     # indicate that the request was a success
        #     data["success"] = True

       
        
        # display = data["predictions"][0]["label"]
    # return type(image)
    return str(jsonData['image'])
    # return render_template("index.html",display=display,imagefile=imagefile)
####################################################################
        # file_path = get_file_path_and_save(request)
        # # data = request.get_json()['data']
        # # data = base64.b64decode(data)
        # # img_data = io.BytesIO(data)
        # img = image.load_img(file_path, target_size=(28, 28), color_mode="grayscale")
        # x = (255-image.img_to_array(img))/255.0
        # x = np.expand_dims(x, axis=0)
        # print('Image Post-processing complete')
        # predictions = model.predict(x)
        # predictions = predictions.reshape(10)
        # print('Predictions completed, llrs = ', predictions)
        # digit_prob_order = np.argsort(-predictions)
        # thresh = 0.99
        # if predictions[digit_prob_order[0]] > thresh:
        #     top_pred_str = ''
        # elif predictions[digit_prob_order[0]] + predictions[digit_prob_order[1]] > thresh:
        #     top_pred_str = '  (Prob = {0:.2f}, Other Predictions: {1} with Prob = {2:.2f})'.format(predictions[digit_prob_order[0]], digit_prob_order[1], predictions[digit_prob_order[1]])
        # else:
        #     top_pred_str = '  (Prob = {0:.2f}, Other Predictions: [{1}, {3}] with Prob = [{2:.2f}, {4:.2f}])'.format(predictions[digit_prob_order[0]], digit_prob_order[1], predictions[digit_prob_order[1]], digit_prob_order[2], predictions[digit_prob_order[2]])
        # digit_str = str(np.argmax(predictions))
        # output_str = digit_str+top_pred_str
        # print('Predicted digit = ', output_str)
        # return output_str

 ###########################################################################       
    # return render_template('results.html', prediction=digit)
    # return jsonify({"prediction": data})


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
