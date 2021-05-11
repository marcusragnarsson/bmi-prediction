# -*- coding: utf-8 -*-


# Commented out IPython magic to ensure Python compatibility.

#!pip install git+https://github.com/rcmalli/keras-vggface.git
#!pip install Keras-Applications
#!pip install mtcnn

import streamlit as st
from PIL import Image
from mtcnn.mtcnn import MTCNN
import numpy as np
from numpy import asarray
from numpy import load
from numpy import expand_dims
from numpy import savez_compressed
import pickle

@st.cache()
def load_vgg():
  # example of creating a face embedding
  from keras_vggface.vggface import VGGFace
  # create a vggface2 model
  model = VGGFace(model='resnet50', include_top=False,input_shape=(224, 224, 3))
  return model

@st.cache()
def load_detector():
  detector = MTCNN()
  return detector

# extract a single face from a given photograph
@st.cache()
def extract_face(pixels, required_size=(224, 224)):
	detector = load_detector()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

@st.cache()
def get_embedding(model, face):
	# scale pixel values
  face = face.astype('float32')
  samples = expand_dims(face, axis=0)
  # prepare the face for the model, e.g. center pixels
  #samples = preprocess_input(samples, version=2)
	# make prediction to get embedding
  yhat = model.predict(samples)
  return yhat[0]

@st.cache()
def load_face_model():

  pkl_filename = '/face_model.pkl'
  with open(pkl_filename, 'rb') as file:
    face_model = pickle.load(file)
  return face_model
  


st.title('Welcome To our BMI Prediction!')
instructions = """
    Please, upload your own image. 
     The image you upload will be fed through the Deep Neural Network in real-time 
    and the output BMI will be displayed to the screen.
    """
st.write(instructions)
image = ''
file = st.file_uploader('Upload An Image',type=['jpg'])
if file:
  model = load_vgg()
  face_model = load_face_model()
  image = Image.open(file)
  img_array = np.array(image)
  extracted = extract_face(img_array)
  face_embedding = get_embedding(model,extracted)
  bmi = face_model.predict(face_embedding[0,:,:])


#bmi = predict(pixels,model)

  st.title("Here is the image you've selected")

  
  st.image(image)
  st.title("Here is the predicted BMI:")

   
  st.write(round(bmi[0], 2))
    

 