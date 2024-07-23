# Vnet-medical-segmentation
A model built using VNet architecture for medical image segmentation.
VNet is a convolutional neural network for segmentation of 3D images,
which can be very useful for segmentation of 3D medical images like
MRI scans or CT scans.
The data used for this model is the LiTS dataset for liver tumor. 
[LiTS-dataset](https://www.kaggle.com/datasets/andrewmvd/lits-png)
The current VNet architecture is based on the architecture provided by
Milletari et al in [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://arxiv.org/abs/1606.04797)
The training size is small, and the depth of the model is low, as I have worked on the images without GPU access.
Each level has two convolution layers with PReLU as the activation function
While UNet uses Maxpooling for downsampling or compressing, VNet uses 3D Convolution with a kernell size of (2,2,2) and stride = 2
For upsampling/expansion, deconvolution was done using Conv3DTranspose function from keras.
The current architecture uses 3 levels of compression followed by 3 levels of expansion. This was chosen for simplicity, but it can go up to 5 levels too.
