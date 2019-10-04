import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
import io
import matplotlib.pyplot as plt
import pickle
import h5py
import sklearn
from sklearn.model_selection import train_test_split
from collections import Counter
import glob
import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import train
from random import shuffle
import imp
import os
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from flask import Flask, request, redirect, url_for, jsonify, render_template
# %matplotlib inline
from werkzeug.utils import secure_filename

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
    # global graph
    imp.reload(train)
    model = train.load_model_from_checkpoint('./models/weights.best.hdf5', six_conv=True)
    # graph = K.get_session().graph
load_model()

def file_predict(image_path, all_perc=False):
    image = cv2.imread(image_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pic = cv2.resize(image, (64,64))
    a = model.predict_proba(pic.reshape(1, 64, 64,3))[0]
    if all_perc:
        print('\n'.join(['{} : {}%'.format(map_characters[i], round(k*100)) for i,k in sorted(enumerate(a), key=lambda x:x[1], reverse=True)]))
    else:
        return map_characters[np.argmax(a)].replace('_',' ').title()



characters = [k.split('/')[2] for k in glob.glob('./characters/*') if len([p for p in glob.glob(k+'/*') if 'edited' in p or 'pic_vid' in p]) > 290]
map_characters = dict(enumerate(characters))
map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel', 
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson', 
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak', 
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}



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
            imagefile= filename

    display = file_predict(filepath)

    return render_template("index.html",display=display,imagefile=imagefile)
 
if __name__ == "__main__":
    app.run(debug=True,port = 8000)