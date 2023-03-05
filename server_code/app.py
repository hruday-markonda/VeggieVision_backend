import os 
import keras
import numpy as np 
import tensorflow as tf
#import pandas as pd
import seaborn as sbn
from keras.models import Sequential, load_model
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import img_to_array, load_img
from flask_cors import CORS

from PIL import Image
#from flask_ngrok import run_with_ngrok

sbn.set()


#from modules import flask
import flask
from flask import *
import requests
from time import time

app = Flask(__name__)
class_names = ['Bean', 'Bitter Gourd', 'Bottle Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']
CORS(app)
#run_with_ngrok(app)


@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        # run model
        print('heloilo')
       
        print(request.files)
        x = request.files['image']
        filename = f"{int(time())}.png"
        x.save(filename)
        pred_img_path = f'/home/ehabaftab/veggie_vision/code/{filename}' #Image path
        pred_img = load_img(pred_img_path, target_size=(256, 256)) #Image loaded and resized to 256x256
        
        
        x = img_to_array(pred_img) #Convert image to matrix with each point being pixel colour


        
        x = np.expand_dims(x, axis=0) #Add an extra dimension so model understands how many rows to process at once in this case 1 row at a time
        #y = x.copy()
        x /= 255. #Standardize pixel vals to be between 0 and 1
        pred_scores = model.predict(x) #Generate prediction scores
        print(np.amax(pred_scores))
        if np.amax(pred_scores) <= 0.97:
            
            data = {
                    'success': False,
                    'error': 'Cannot identify Vegetable'
                    }

            return data
        pred_class = class_names[int(np.argmax(pred_scores))] #Pick the class
        print("Your vegetable is {}".format(pred_class)) #Print class
        data = {
                'success' : True,
                'vegetable' : pred_class
        }
       
        return data  
        
    if request.method == 'GET':
        return flask.Response(status=200)

# Starting the python applicaiton
if __name__ == '__main__':

    # Specify Model Path and load model
    model_path = '/home/ehabaftab/veggie_vision/whatgetable_ResNet50V2.h5'
    model = load_model(model_path)
    # Note, you're going to have to change the PORT number
    #run_with_ngrok(app)
    app.run(host='0.0.0.0' ,port=5128)
    #app.run(port=5128)
