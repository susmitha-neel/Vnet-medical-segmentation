import streamlit as st
import numpy as np
import nibabel as nib
import h5py
import pydicom
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model, model_from_json

import scipy.misc
import scipy.ndimage

#path to model/train data/predictions
save_dir = "C:/Users/dell/Downloads/"

# Load your model
model = load_model("C:/Users/dell/Downloads/model.hdf5")

#weights for the model
weights_dir = save_dir+"vnet_weights." + "weights.h5"

#load weights to the model
model.load_weights(weights_dir)


#  load  images
def load_h5(file):
    h5_image = h5py.File(file,'r')
    data = h5_image.get('data')
    #truth = h5_image.get('truth')

    return data


def predict(test, model):
    predictions = []
    m = test.shape[0]
    st.write('Starting predictions:')
    st.write("0/%i (0%%)" % m)

    for i in range(m):
        image = test[i][np.newaxis, :, :, :]
        prediction = model.predict(image, steps=1)
        prediction = np.squeeze(prediction)
        predictions.append(prediction)
        st.write("%i/%i (%i%%)" % (i + 1, m, ((i + 1) / m * 100)))

    predictions = np.array(predictions)

    st.write('Predictions obtained with shape:')
    st.write(predictions.shape)
    return predictions


def write_predictions(predicitons, path):
    h5f = h5py.File(path, 'w')
    h5f.create_dataset(name='predictions', data=predicitons)
    h5f.close()
    st.write('Predictions saved to ' + path)


# app
st.title('3D Medical Image Segmentation')

uploaded_h5_file = st.file_uploader('Choose a  hd5 file', type = 'h5')

if uploaded_h5_file is not None:
    x_test = load_h5(uploaded_h5_file)
else:
    st.error('Unsupported file type')

if st.button('Predict Segmentation'):
    predictions = predict(x_test, model)
    predictions = np.array(np.split(predictions, x_test.shape[0], axis=0))
    write_predictions(predictions, save_dir+"predictions_vnet_from_app.h5")

