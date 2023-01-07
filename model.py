# Importing required libs
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from werkzeug.utils import secure_filename

map_characters = {0: 'abraham_grampa_simpson', 1: 'apu_nahasapeemapetilon', 2: 'bart_simpson', 
        3: 'charles_montgomery_burns', 4: 'chief_wiggum', 5: 'comic_book_guy', 6: 'edna_krabappel', 
        7: 'homer_simpson', 8: 'kent_brockman', 9: 'krusty_the_clown', 10: 'lisa_simpson', 
        11: 'marge_simpson', 12: 'milhouse_van_houten', 13: 'moe_szyslak', 
        14: 'ned_flanders', 15: 'nelson_muntz', 16: 'principal_skinner', 17: 'sideshow_bob'}


# Loading model
model = load_model('models/model.h5')

def get_file_path_and_save(request):
    f = request.files['file']
    file_path = 'images/'+secure_filename(f.filename)
    f.save(file_path)
    return file_path

def center_crop(img, dim):
	"""Returns center cropped image
	Args:
	img: image to be center cropped
	dim: dimensions (width, height) to be cropped
	"""
	width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension
	crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
	crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
	mid_x, mid_y = int(width/2), int(height/2)
	cw2, ch2 = int(crop_width/2), int(crop_height/2) 
	crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
	return crop_img


# Preparing and pre-processing the image
def preprocess_img(request):
    #f = request.files['file']
    file_path = get_file_path_and_save(request)
    image = cv2.imread(file_path)
    cropped_img = center_crop(image, image.shape)
    pic = cv2.resize(cropped_img, (64,64))
    img_reshape = pic.reshape(1, 64, 64, 3)
    return img_reshape


# Predicting function
def predict_result(predict):
    pred = model.predict(predict)
    return map_characters[np.argmax(pred[0])].replace('_',' ').title()