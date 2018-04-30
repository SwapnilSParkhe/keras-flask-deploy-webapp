from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from pickle import load

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.wsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
Model_path = 'models/best_model_weights.h5'
Tokenizer_path= 'models/tokenizer.pkl'

# Load your trained model and tokenizer
model = load_model(Model_path)
tokenizer=load(open(Tokenizer_path,'rb'))

print('Model loaded. Check http://127.0.0.1:5000/')

###########################################################################
#Extracting image features first
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
def ImageFeature_Extractor(ImageName): 
    #Re-structuring the VGG model as per requirements
    model_=VGG16(weights="imagenet")  #Loading the model
    model_.layers.pop()  #Restructing model (retain penultimate FCC-4096)
    model_=Model(inputs=model_.inputs, outputs=model_.layers[-1].output)

    #Extracting features from image (jpg) using restructured model   
    image=load_img(ImageName,target_size=(224,224)) #loading img aptly
    image=img_to_array(image)   #converting PIL image to array
    image=image.reshape((1,image.shape[0],image.shape[1],image.shape[2]))  
    image=preprocess_input(image)   # preprocessing of img for VGG
    image_feature=model_.predict(image,verbose=0)   #getting img features

    return image_feature

#Mapping an integer prediction back to a word
#Note: Using the same tokeniser used for train data
def intID_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

#Generating a desc for an image using trained model
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
def generate_desc(model, tokenizer, image, max_length):
    in_text='startseq'   #seeding the generation process
    for i in range(max_length):
        seq=tokenizer.texts_to_sequences([in_text])[0] #encoding txt2int
        seq=pad_sequences([seq], maxlen=max_length) #padding seq
        pred=model.predict([image, seq], verbose=0) #predict next word
        pred=argmax(pred)   #prob to integer ID conversion
        word=intID_to_word(pred, tokenizer) #intID to word mapping
        if word is None:
            break   #stop if cant map word
        in_text += ' ' + word  #append as input to generate next word
        if word=='endseq':
            break   #stop if end of seq
    return in_text  
#########################################################################  

def model_predict(img_path, model):
    img_feat=ImageFeature_Extractor(img_path) #Image Features extracted
    max_length=33
    caption=generate_desc(model, tokenizer, img_feat, max_length)
    
    def Cap_first_word(string_in): #Formating result output
      tokens=string_in.split()
      tokens_1=tokens[1:len(tokens)-1]
      tokens_2=" ".join(tokens_1)
      string_out=tokens_2.capitalize()
      return string_out

    caption_=Cap_first_word(caption)
    return caption_
  
@app.route('/', methods=['GET'])
def index():
    # Main page template
    return render_template('index.html')
  
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        #captioning
        caption=model_predict(file_path, model)
        return caption
    return None
    

if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
