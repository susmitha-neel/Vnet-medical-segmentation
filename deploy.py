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

#preprocess images
INPUT_SIZE = 128  # Input feature width/height
INPUT_DEPTH = 64  # Input depth

MIN_HU = -160  # Min HU Value
MAX_HU = 240  # Max HU Value







def hu_window(image):
    image[image < MIN_HU] = MIN_HU
    image[image > MAX_HU] = MAX_HU
    image = (image - MIN_HU) / (MAX_HU - MIN_HU)
    image = image.astype("float32")
    return image


def scale_volume(volume, img_depth=INPUT_DEPTH, img_px_size=INPUT_SIZE, hu_value=True):
    if hu_value:
        volume = hu_window(volume)

    size_scale_factor = img_px_size / volume.shape[0]
    depth_scale_factor = img_depth / volume.shape[-1]

    volume = scipy.ndimage.interpolation.rotate(volume, 90, reshape=False)

    vol_zoom = scipy.ndimage.interpolation.zoom(volume, [size_scale_factor, size_scale_factor,
                                                         depth_scale_factor], order=1)

    vol_zoom[vol_zoom < 0] = 0
    vol_zoom[vol_zoom > 1] = 1
    return vol_zoom


def scale_segmentation(segmentation, img_depth=INPUT_DEPTH, img_px_size=INPUT_SIZE):
    size_scale_factor = img_px_size / segmentation.shape[0]
    depth_scale_factor = img_depth / segmentation.shape[-1]

    segmentation = scipy.ndimage.interpolation.rotate(segmentation, 90, reshape=False)

    # Nearest neighbour is used to mantain classes discrete
    seg_zoom = scipy.ndimage.interpolation.zoom(segmentation, [size_scale_factor, size_scale_factor,
                                                               depth_scale_factor], order=0)

    return seg_zoom

def divide_segmentation(segmentation):
    layer1 = np.copy(segmentation)
    layer2 = np.copy(segmentation)
    background = np.copy(segmentation)

    layer1[layer1==2] = 1
    layer2[layer2==1] = 0
    layer2[layer2 > 0] = 1
    background[background==1] = 5
    background[background == 0] = 1
    background[background>1] = 0
    # np.concatenate((layer1, layer2, background), axis=-1)
    # return layer1, layer2, background
    # a = np.concatenate((layer1, layer2, background), axis=-1)
    # return a
    return layer1  # liver



def get_data(vol_dir, seg_dir,crop=False):
    volume = nib.load(vol_dir).get_data()
    segmentation = nib.load(seg_dir).get_data()

    if crop:
        aux = []
        for i in range(segmentation.shape[2]):
            if np.sum(segmentation[:, :, i]) > 0:
                aux.append(i)

        volume = volume[:, :, (np.min(aux)-1):(np.max(aux)+1)]
        segmentation = segmentation[:, :, (np.min(aux)-1):(np.max(aux)+1)]

    return volume, segmentation

def create_dataset(volume_file,segment_file, px_size=INPUT_SIZE, slice_count=INPUT_DEPTH, crop=False):
    """Returns dataset with shape (m, z, x, y, n)"""

    segmentations = []
    volumes = []


    # m = len(volumes)

    #if crop:
        #print("Creating Cropped Data Set:")
    #else:
        #print("Creating Data Set:")

    slices = []

    volume, segmentation = get_data(volume_file, segment_file, crop)

    if volume.shape[0] == 0 or volume.shape[1] == 0 or volume.shape[2] == 0:
        st.write("Image is missing information")
    else:
        volume = scale_volume(volume, slice_count, px_size)
    if segmentation.shape[0] == 0 or segmentation.shape[1] == 0 or segmentation.shape[2] == 0:
        st.write("Image is missing information")
    else:
        segmentation = scale_segmentation(segmentation, slice_count, px_size)
    slices.append([volume, segmentation])


    dataset = np.array(slices)

    #print("Dataset finished with shape:")
    #print(dataset.shape)

    return dataset

#Store image as h5 file
def write_dataset(data_set, path):
    h5f = h5py.File(path, 'w')
    h5f.create_dataset('data', data=np.expand_dims(data_set[:, 0, :, :, :], -1))
    truth = np.expand_dims(data_set[:, 1, :, :, :], -1)
    truth = divide_segmentation(truth)
    h5f.create_dataset('truth', data=truth)
    h5f.close()
    #print("Dataset saved @ %s" % path)




















#  load  images
def load_h5(file):
    h5_image = h5py.File(file,'r')
    data = h5_image.get('data')
    truth = h5_image.get('truth')

    return data, truth

#load nifti image
def load_nifti(file):
    nifti_image = nib.load(file)
    image = nifti_image.get_fdata()
    return image



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
vol_image = None
segment_image = None

uploaded_volume_file = st.file_uploader('Choose a nifti volume file', type='nii')
uploaded_segmentation_file = st.file_uploader('Choose a nifti segment file',type = 'nii')
uploaded_h5_file = st.file_uploader('Choose a  hd5 file', type = 'h5')

if uploaded_volume_file is not None:
    file_type = uploaded_volume_file.name[0]    
    if file_type == 'v':
        vol_image = load_nifti(uploaded_volume_file)
    else:
        st.error("Not a volume file")


if uploaded_segmentation_file is not None:
    file_type = uploaded_segmentation_file.name[0]
    if file_type == 's':
        segment_image = load_nifti(uploaded_segmentation_file)
    else:
        st.error("Not a segmentation file")


if vol_image is not None and segment_image is not None:
    test_set = create_dataset(vol_image, segment_image,crop=True)
    nifti_h5_file = write_dataset(test_set, save_dir + "test_data.h5")

if uploaded_h5_file is not None:
    x_test, ytest = load_h5(uploaded_h5_file)
else:
    st.error('Unsupported file type')

if st.button('Predict Segmentation'):

    predictions = predict(x_test, model)
    predictions = np.array(np.split(predictions, x_test.shape[0], axis=0))
    write_predictions(predictions, save_dir+"predictions_vnet_from_app.h5")

