# -*- coding: utf-8 -*-
"""Fullnet_(2) (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pDJBksNiUETs6Z6a-Dkv3ltCVUtmlQ3l

#Все
"""

import numpy as np
import random
from tensorflow.keras.models import load_model, Model  # Импортируем модели keras: Model
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation, SpatialDropout2D, MaxPooling2D, \
    AveragePooling2D, Conv2D, BatchNormalization, Rescaling  # Импортируем стандартные слои keras
from tensorflow.keras import backend as K  # Импортируем модуль backend keras'а
from tensorflow.keras.optimizers import Adam  # Импортируем оптимизатор Adam
from tensorflow.keras import \
    utils  # Импортируем модуль utils библиотеки tensorflow.keras для получения OHE-представления
from keras import regularizers
from keras.callbacks import Callback
import os
import tensorflow as tf
import cv2
import gc

GrayScaled = True

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# random.seed(1)

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


def unetWithMask(num_classes=2, input_shape=(128, 128, 1)):
    img_input = Input(input_shape)  # Создаем входной слой с размерностью input_shape

    x = Rescaling(1/255)(img_input)
    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(x)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_1_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_1_out

    block_1_out_mask = Conv2D(64, (1, 1), padding='same')(
        block_1_out)  # Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_1_out_mask

    x = MaxPooling2D()(block_1_out)  # Добавляем слой MaxPooling2D

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_2_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_2_out

    block_2_out_mask = Conv2D(128, (1, 1), padding='same')(
        block_2_out)  # Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_2_out_mask

    x = MaxPooling2D()(block_2_out)  # Добавляем слой MaxPooling2D

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_3_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_3_out

    block_3_out_mask = Conv2D(256, (1, 1), padding='same')(
        block_3_out)  # Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_3_out_mask

    x = MaxPooling2D()(block_3_out)  # Добавляем слой MaxPooling2D

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)  # Добавляем Conv2D-слой с 256-нейронами
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
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(
        x)  # Добавляем слой Conv2DTranspose с 256 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_3_out,
                     block_3_out_mask])  # Объединем текущий слой со слоем block_3_out и слоем-маской block_3_out_mask
    x = Conv2D(256, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 256 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 256 нейронами
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


def GetBatch(X, Y):
    while (True):
        StartBatch = 0

        while StartBatch < len(X):
            EndBatch = StartBatch + 16

            InputX = X[StartBatch:EndBatch]
            InputY = Y[StartBatch:EndBatch]

            yield InputX, InputY

            StartBatch = EndBatch

            # gc.collect()

def Read(Name = 'train2', Sz = 50000, TestSz = 16 * 64):
    Data = np.load(rf'{CrPath}numpy/{Name}.npy')

    YSum = np.sum(Data[:-TestSz, 1].reshape((len(Data)-TestSz, Data.shape[-1] * Data.shape[-2])), 1)

    Ind = YSum.argsort()[-Sz:]
    YSum = None
    #gc.collect()
    IndInd = np.arange(Sz)
    np.random.shuffle(IndInd)
    Ind = Ind[IndInd]
    IndInd = None

    np.save('sorted.npy', Data[Ind])
    np.save('test.npy', Data[-TestSz:])


Read()
gc.collect()

Data = np.load('sorted.npy')
Test = np.load('test.npy')

InputX = Data[:, 0]
XShape = list(InputX.shape)
XShape.append(1)
InputX = InputX.reshape(XShape)

InputY = Data[:, 1]
YShape = list(InputY.shape)
YShape.append(1)
InputY = InputY.reshape(YShape)
InputY1 = 1 - InputY
InputY = np.concatenate([InputY, InputY1], 3)
InputY1 = None

print(0)

gc.collect()

TestX = Test[:, 0]
XShape = list(TestX.shape)
XShape.append(1)
TestX = TestX.reshape(XShape)

TestY = Test[:, 1]
Test = None
YShape = list(TestY.shape)
YShape.append(1)
TestY = TestY.reshape(YShape)
TestY1 = 1 - TestY

TestY = np.concatenate([TestY, TestY1], 3)
TestY1 = None

gc.collect()

print(3)

try:
    model = load_model(rf'{CrPath}cr.h5')
    # model.load_weights(rf'{CrPath}cr.npy')
    print('Прочитаны веса')

except:
    try:
        model = load_model(rf'{CrPath}best.h5')
        print('Прочитана модель')
    except:
        model = unetWithMask()
        print('Модель создана со случайными весами')

# W = np.load(rf'{CrPath}cr.npy', allow_pickle=True)

# model.set_weights(W)

import gc

Best = 0
Index = 0
G = GetBatch(InputX, InputY)

print('Поехали')

while True:
    history = model.fit(G, epochs=1, steps_per_epoch=1, verbose=1, use_multiprocessing=True)

    Res = model.predict(TestX, steps=64, verbose=True)[:, :, :, 0:1]
    er = dice_coef_np(Res, TestY)

    Res = None

    if er > Best:
        Best = er
        model.save(rf'{CrPath}best.h5')
        print('Сохранена лучшая модель. ', Index, er, Best)
    else:
        print(Index, er, Best)

    model.save(rf'{CrPath}cr.h5')
    Index += 1