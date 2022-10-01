from UNetGenetic import TUNetGenetic
import numpy as np
import random
from NetFromHromosom import TNNHromosom, BinClassification, dice_coef
from tensorflow.keras.models import Model, load_model # Импортируем модели keras: Model
from tensorflow.keras.layers import Input, Rescaling, Conv2DTranspose, concatenate, Activation, SpatialDropout2D, MaxPooling2D, AveragePooling2D, Conv2D, BatchNormalization # Импортируем стандартные слои keras
from tensorflow.keras import backend as K # Импортируем модуль backend keras'а
from tensorflow.keras.optimizers import Adam # Импортируем оптимизатор Adam
from tensorflow.keras import utils # Импортируем модуль utils библиотеки tensorflow.keras для получения OHE-представления
from keras import regularizers
from keras.callbacks import Callback
import tensorflow as tf
import os
from tqdm.auto import tqdm
import glob
import cv2
import albumentations as A
from augmentation import MakeAug
from PrepareImg import EyeDataset
import gc
from tqdm.auto import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

Colab = True
try:
    from google.colab import drive
except:
    Colab = False

if Colab:
    from google.colab import drive

    # Подключаем Google drive
    drive.mount('/content/drive')
    CrPath = "/content/drive/MyDrive/Henetic/"
else:
    Acer = not os.path.exists("E:/w/Diplom/")
    CrPath = "C:/w/Hacatons/Vladik/" if Acer else "E:/w/Hacatons/Vladik/"

def unetWithMask(num_classes=2, input_shape=(512, 512, 1)):
    img_input = Input(input_shape)  # Создаем входной слой с размерностью input_shape

    x = Rescaling(1 / 255)(img_input)
    # Block 1
    x = Conv2D(80, (3, 3), padding='same', name='block1_conv1')(x)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(80, (3, 3), padding='same', name='block1_conv2')(x)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_1_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_1_out

    block_1_out_mask = Conv2D(80, (1, 1), padding='same')(
        block_1_out)  # Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_1_out_mask

    x = MaxPooling2D()(block_1_out)  # Добавляем слой MaxPooling2D

    # Block 2
    x = Conv2D(140, (3, 3), padding='same', name='block2_conv1')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(140, (3, 3), padding='same', name='block2_conv2')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_2_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_2_out

    block_2_out_mask = Conv2D(140, (1, 1), padding='same')(
        block_2_out)  # Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_2_out_mask

    x = MaxPooling2D()(block_2_out)  # Добавляем слой MaxPooling2D

    # Block 3
    x = Conv2D(300, (3, 3), padding='same', name='block3_conv1')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(300, (3, 3), padding='same', name='block3_conv2')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(300, (3, 3), padding='same', name='block3_conv3')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_3_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_3_out

    block_3_out_mask = Conv2D(300, (1, 1), padding='same')(
        block_3_out)  # Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_3_out_mask

    x = MaxPooling2D()(block_3_out)  # Добавляем слой MaxPooling2D

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_4_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_4_out

    block_4_out_mask = Conv2D(512, (1, 1), padding='same')(
        block_4_out)  # Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_4_out_mask

    x = MaxPooling2D()(block_4_out)  # Добавляем слой MaxPooling2D

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 1
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(
        x)  # Добавляем слой Conv2DTranspose с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_4_out,
                     block_4_out_mask])  # Объединем текущий слой со слоем block_4_out и слоем-маской block_4_out_mask
    x = Conv2D(512, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 2
    x = Conv2DTranspose(300, (2, 2), strides=(2, 2), padding='same')(
        x)  # Добавляем слой Conv2DTranspose с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_3_out,
                     block_3_out_mask])  # Объединем текущий слой со слоем block_3_out и слоем-маской block_3_out_mask
    x = Conv2D(300, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(300, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 3
    x = Conv2DTranspose(140, (2, 2), strides=(2, 2), padding='same')(
        x)  # Добавляем слой Conv2DTranspose с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_2_out,
                     block_2_out_mask])  # Объединем текущий слой со слоем block_2_out и слоем-маской block_2_out_mask
    x = Conv2D(140, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(140, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 4
    x = Conv2DTranspose(80, (2, 2), strides=(2, 2), padding='same')(x)  # Добавляем слой Conv2DTranspose с 64 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_1_out,
                     block_1_out_mask])  # Объединем текущий слой со слоем block_1_out и слоем-маской block_1_out_mask
    x = Conv2D(80, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(80, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(
        x)  # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов

    model = Model(img_input, x)  # Создаем модель с входом 'img_input' и выходом 'x'

    # Компилируем модель
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy')

    return model  # Возвращаем сформированную модель


def unetWithMask6(num_classes=2, input_shape=(512, 512, 1)):
    img_input = Input(input_shape)  # Создаем входной слой с размерностью input_shape

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_1_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_1_out

    block_1_out_mask = Conv2D(64, (1, 1), padding='same')(
        block_1_out)  # 512 Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_1_out_mask

    x = MaxPooling2D()(block_1_out)  # Добавляем слой MaxPooling2D

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_2_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_2_out

    block_2_out_mask = Conv2D(128, (1, 1), padding='same')(
        block_2_out)  #128 Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_2_out_mask

    x = MaxPooling2D()(block_2_out)  # Добавляем слой MaxPooling2D

    # Block 3
    x = Conv2D(512, (3, 3), padding='same', name='block3_conv1')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block3_conv2')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block3_conv3')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_3_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_3_out

    block_3_out_mask = Conv2D(512, (1, 1), padding='same')(
        block_3_out)  #64 Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_3_out_mask

    x = MaxPooling2D()(block_3_out)  # Добавляем слой MaxPooling2D

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_4_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_3_out

    block_4_out_mask = Conv2D(512, (1, 1), padding='same')(
        block_4_out)  # Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_3_out_mask

    x = MaxPooling2D()(block_4_out)  # Добавляем слой MaxPooling2D

    # Block 5
    x = Conv2D(1024, (3, 3), padding='same', name='block5_conv1')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(1024, (3, 3), padding='same', name='block5_conv2')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(1024, (3, 3), padding='same', name='block5_conv3')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_5_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_4_out

    block_5_out_mask = Conv2D(1024, (1, 1), padding='same')(
        block_5_out)  # Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_4_out_mask

    x = MaxPooling2D()(block_5_out)  # Добавляем слой MaxPooling2D

    # Block 6
    x = Conv2D(1024, (3, 3), padding='same', name='block6_conv1')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(1024, (3, 3), padding='same', name='block6_conv2')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(1024, (3, 3), padding='same', name='block6_conv3')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 0
    x = Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(
        x)  # Добавляем слой Conv2DTranspose с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_5_out,
                     block_5_out_mask])  # Объединем текущий слой со слоем block_4_out и слоем-маской block_4_out_mask
    x = Conv2D(1024, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(1024, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 1
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(
        x)  # Добавляем слой Conv2DTranspose с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_4_out,
                     block_4_out_mask])  # Объединем текущий слой со слоем block_4_out и слоем-маской block_4_out_mask
    x = Conv2D(512, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 2
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(
        x)  # Добавляем слой Conv2DTranspose с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_3_out,
                     block_3_out_mask])  # Объединем текущий слой со слоем block_3_out и слоем-маской block_3_out_mask
    x = Conv2D(512, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(
        x)  # Добавляем слой Conv2DTranspose с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_2_out,
                     block_2_out_mask])  # Объединем текущий слой со слоем block_2_out и слоем-маской block_2_out_mask
    x = Conv2D(128, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)  # Добавляем слой Conv2DTranspose с 64 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_1_out,
                     block_1_out_mask])  # Объединем текущий слой со слоем block_1_out и слоем-маской block_1_out_mask
    x = Conv2D(64, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(
        x)  # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов

    model = Model(img_input, x)  # Создаем модель с входом 'img_input' и выходом 'x'

    # Компилируем модель
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy')

    return model  # Возвращаем сформированную модель

def dice_coef_np(y_true, y_pred):
    return (2. * np.sum(y_true * y_pred) + 1.) / (np.sum(y_true) + np.sum(y_pred) + 1.)

X = []
Y = []
InBatch = 0
Au = 0

def GFun(dataset, BatchSz = 16, AuSteps = 10, Test = False):
    global InBatch
    global X
    global Y
    global InBatch
    global Au

    while (True):
        #for Au in range(AuSteps):
        while Au < AuSteps:
            for sample in dataset:

                # if Step < FileIndex*50: # аккуратненько чтоб не вникать в тонкости устройства dataset пропускаем уже сделанные позиции (использую при обрыве выполнения)
                #    Step+= 1
                # continue

                m = sample["mask"][:, :, 1]
                Img = sample['image']

                Sz = Img.shape[:2]

                Img = cv2.resize(Img, (512, 512))
                Mask = cv2.resize(m, (512, 512))

                if Au > 0 or Test:
                    Img, Mask = MakeAug(Img, Mask, verbose=True)

                Au += 1

                InBatch -= 1

                if InBatch > 0:
                    X.append(Img.reshape((1, 512, 512, 1)))
                    Y.append(Mask.reshape((1, 512, 512, 1)))
                else:
                    return (np.concatenate(X), np.concatenate(Y))
                    X = []
                    Y = []
                    InBatch = BatchSz

                Img = None
                Mask = None

                gc.collect()


def Generate(dataset, BatchSz = 16, AuSteps = 10, Test = False):
    InBatch = BatchSz
    random.seed(11)

    X = []
    Y = []

    while(True):
        for Au in range(AuSteps):
            for sample in dataset:

                # if Step < FileIndex*50: # аккуратненько чтоб не вникать в тонкости устройства dataset пропускаем уже сделанные позиции (использую при обрыве выполнения)
                #    Step+= 1
                # continue

                m = sample["mask"][:, :, 1]
                Img = sample['image']

                Sz = Img.shape[:2]

                Img = cv2.resize(Img, (512, 512))
                Mask = cv2.resize(m, (512, 512))

                if Au > 0 or Test:
                    Img, Mask = MakeAug(Img, Mask, verbose= False)

                if InBatch > 0:
                    Mask = Mask.reshape((1, 512, 512, 1))
                    Mask1 = 1 - Mask

                    Mask = np.concatenate([Mask, Mask1], 3)
                    X.append(Img.reshape((1, 512, 512, 1)))
                    Y.append(Mask)

                    InBatch -= 1
                else:
                    yield (np.concatenate(X), np.concatenate(Y))
                    X = []
                    Y = []
                    InBatch = BatchSz

                Img = None
                Mask = None

                gc.collect()

try:
    model = load_model(rf'{CrPath}CrFull.h5')
    # model.load_weights(rf'{CrPath}cr.npy')
    print('Прочитаны веса')

except:
    try:
        model = load_model(rf'{CrPath}BestFull.h5')
        print('Прочитана модель')
    except:
        model = unetWithMask()
        print('Модель создана со случайными весами')

Best = 0
Index = 0
dataset = EyeDataset("train", ImgReadMode=cv2.IMREAD_GRAYSCALE)

G = Generate(dataset)
Test = Generate(dataset, Test = True)
print('Поехали')

#TestX = [0]*64
#TestY = [0]*64

#X, Y = next(G)

print('Генерируем Test набор')
index = 0



TestX = np.load(fr'{CrPath}TestX.npy')
TestY = np.load(fr'{CrPath}TestY.npy')

'''
try:
    TestX = np.load('TestX.npy')
    TestY = np.load('TestY.npy')
except:
    for index in tqdm(range(64)):
        X, Y = next(Test)

        TestX[index] = X
        TestY[index] = Y

    try:
        TestX = np.concatenate(TestX)
        TestY = np.concatenate(TestY)
    except:
        print('')

    np.save('TestX', TestX)
    np.save('TestY', TestY)
'''

while True:
    X, Y = next(G)
    history = model.fit(X, Y, epochs=1, batch_size=16, verbose=1)

    Res = model.predict(TestX, steps=64, verbose=True)[:, :, :, 0:1]
    er = dice_coef_np(Res, TestY)

    Res = None

    if er > Best:
        Best = er
        model.save(rf'{CrPath}BestFull.h5')
        print('Сохранена лучшая модель. ', Index, er, Best)
    else:
        print(Index, er, Best)

    model.save(rf'{CrPath}CrFull.h5')
    Index += 1