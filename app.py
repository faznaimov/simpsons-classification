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
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import train
from random import shuffle
import imp
import os
from PIL import Image, ImageFont, ImageDraw, ImageEnhance


from PIL import Image
import base64
import io
import numpy as np
# import torch

app = Flask(__name__)


characters = [k.split('/')[2] for k in glob.glob('./characters/*') if len([p for p in glob.glob(k+'/*') if 'edited' in p or 'pic_vid' in p]) > 290]
map_characters = dict(enumerate(characters))
map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel', 
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson', 
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak', 
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}


def readb64(test_image_base64_encoded):
    base64_decoded = base64.b64decode(test_image_base64_encoded)
    image = Image.open(io.BytesIO(base64_decoded))

    return image


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
       
       # train the model
        imp.reload(train)
        model = train.load_model_from_checkpoint('./models/weights.best.hdf5', six_conv=True)
        
        # Get data from frontend
        jsonData = request.json

        # Decode base64 image using readb64 function
        decodedImage = readb64(jsonData['image'])

        # convert to np array
        image_array = np.asarray(decodedImage, dtype="uint8")

        # Resize and make prediction
        pic = cv2.resize(image_array, (64,64))
        a = model.predict_proba(pic.reshape(1, 64, 64,3))[0]
        
        # return prediction to front end
        return map_characters[np.argmax(a)].replace('_',' ').title()



if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)
