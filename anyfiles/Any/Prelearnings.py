import numpy

Colab = False
Acer = False
ver = 2
ModelFileName = "weights1 0001.hdf5"
MaxEpoch = 200

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, \
    ZeroPadding2D, GlobalAveragePooling2D
from keras.layers import add, Flatten, Activation
from tensorflow.keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from keras.applications import efficientnet_v2
from tensorflow.keras.applications.resnet50 import ResNet50
from pathlib import Path
import glob, os
from PIL import Image
import pickle
import gc


Colab = True
try:
    from google.colab import drive
except:
    Colab = False

if Colab:
    from google.colab import drive

    # Подключаем Google drive
    drive.mount('/content/drive')
    CrPath = "/content/drive/MyDrive/"
else:
    Acer = not os.path.exists("E:/w/Diplom/")
    CrPath = "C:/w/Hacatons/Vladik/" if Acer else "E:/w/Hacatons/Vladik/"

ENet = efficientnet_v2.EfficientNetV2L(weights='imagenet', include_top=False)

ENet.get_layer('block2f_add').output
ENet.get_layer('block2f_add').output
ENet.get_layer('block2f_add').output
ENet.get_layer('block2f_add').output
ENet.get_layer('block2f_add').output
ENet.get_layer('block2f_add').output
ENet.get_layer('block2f_add').output
ENet.get_layer('block2f_add').output
block1c_add
block2f_add
block3f_add
block4i_add
block5r_add
block6y_add
block7g_add

ENem
a = 0