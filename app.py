import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2
# import urllib

from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from flask import Flask, render_template, request, url_for, jsonify, redirect


app = Flask(__name__)

map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel', 
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson', 
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak', 
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}

# CNN Keras model with 6 convolutions.
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=[64,64,3]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same')) 
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(map_characters), activation='softmax'))

    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

# def url_to_image(url):
#     resp = urllib.request.urlopen(url)
#     image = np.asarray(bytearray(resp.read()), dtype="uint8")
#     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#     return image

# def url_predict(url):
#     image = url_to_image(url)
#     pic = cv2.resize(image, (64,64))
#     a = model.predict_proba(pic.reshape(1, 64, 64,3))[0]
#     return a

# load weights into new model
global model
model = create_model()
model.load_weights('./models/model.h5') 

global graph
graph = tf.get_default_graph()

def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']
    
    # Save the file to ./uploads
    # basepath = os.path.dirname(__file__)
    # file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
    file_path = 'models/'+secure_filename(f.filename)
    f.save(file_path)
    return file_path

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':
        file_path = get_file_path_and_save(request)
        image = cv2.imread(file_path)
        pic = cv2.resize(image, (64,64))
        
        with graph.as_default():
            a = model.predict_proba(pic.reshape(1, 64, 64,3))[0]
        
        output_str = map_characters[np.argmax(a)].replace('_',' ').title()
        
        return output_str

@app.route('/classify_url', methods=['GET'])
def classify_url():
        
        with graph.as_default():
            a = model.predict_proba(pic.reshape(1, 64, 64,3))[0]
        
        output_str = map_characters[np.argmax(a)].replace('_',' ').title()

        return output_str

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True, threaded=False)
