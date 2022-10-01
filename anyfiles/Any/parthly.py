import numpy as np
import random
from tensorflow.keras.models import Model, load_model  # Импортируем модели keras: Model
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation, SpatialDropout2D, MaxPooling2D, \
    AveragePooling2D, Conv2D, BatchNormalization  # Импортируем стандартные слои keras
from tensorflow.keras import backend as K  # Импортируем модуль backend keras'а
from tensorflow.keras.optimizers import Adam  # Импортируем оптимизатор Adam
from tensorflow.keras import \
    utils  # Импортируем модуль utils библиотеки tensorflow.keras для получения OHE-представления
from keras import regularizers
from keras.callbacks import Callback
import os
import tensorflow as tf
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


# a = np.array([ [11,12,13],[14,15,16],[17,18,19]])
# = np.array([ [1,2],[3,4]])

# a[0:2,0:2] = b
# print(a)


def unetWithMask(num_classes=2, input_shape=(128, 128, 1)):
    img_input = Input(input_shape)  # Создаем входной слой с размерностью input_shape

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)  # Добавляем Conv2D-слой с 64-нейронами
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
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])

    return model  # Возвращаем сформированную модель


def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


Data = np.load(rf'{CrPath}numpy/train2.npy')
print(Data.shape)
Shape = Data.shape
# Data = Data.reshape( (Shape[0]*Shape[1]*Shape[2], Shape[3], Shape[4], Shape[5]))
# np.random.shuffle(Data)
# np.save(rf'{CrPath}numpy/train2.npy', Data)
# print('Сохранено')
NTest = len(Data) // 10
TestPos = len(Data) - Теуые
xlen = 5000 + NTest
print(xlen - NTest)

Data = Data[:xlen]

InputX0 = Data[NTest:xlen, 0]
XShape = list(InputX0.shape)
XShape.append(1)
InputX0 = InputX0.reshape(XShape)

InputY0 = Data[NTest:xlen, 1]
YShape = list(InputY0.shape)
YShape.append(1 if GrayScaled else 3)
InputY0 = InputY0.reshape(YShape)

# InputX1 = Data[xlen//2:-NTest, 0]/255
# XShape = list(InputX1.shape)
# XShape.append(1)
# InputX1 = InputX1.reshape(XShape)

print(1)

# InputY1 = Data[xlen//2:-NTest, 1]
# YShape = list(InputY1.shape)
# YShape.append(1 if GrayScaled else 3)
# InputY1 = InputY1.reshape(YShape)

gc.collect()
print(2)

TestX = Data[:NTest] / 255
TestXShape = list(TestX.shape)
TestXShape.append(1)
TestX = TestX.reshape(TestXShape)

print(3)

TestY = Data[:NTest]
TestYShape = list(TestY.shape)
TestYShape.append(1 if GrayScaled else 3)
TestY = TestY.reshape(TestYShape)
TestY1 = 1 - TestY
TestY = np.concatenate([TestY, TestY1], 3)

print(4)
Data = None
gc.collect()

# YSum = np.sum(InputY0.reshape((InputY0.shape[0], InputY0.shape[1] * InputY0.shape[2])), 1)
# Ind = YSum.argsort()
# InputX0 = InputX0[Ind]/255
# InputY0 = InputY0[Ind]
# InputY01 = 1 - InputY0
InputY0 = np.concatenate([InputY0, 1 - InputY0], 3)

print(5)

# YSum = np.sum(InputY1.reshape((InputY1.shape[0], InputY1.shape[1] * InputY1.shape[2])), 1)
# Ind = YSum.argsort()
# InputX1 = InputX1[Ind]/255
# InputY1 = InputY1[Ind]
# InputY11 = 1 - InputY1
# InputY1 = np.concatenate([InputY1, 1 - InputY1], 3)

print('Ready0')


class THromosomCallback(Callback):
    def __init__(self):
        self.State = [0, 0]  # лучшее значение, его позиция
        self.BestW = None

    def on_epoch_end(self, epoch, logs=None):
        if (logs['dice_coef'] == None) or (logs['dice_coef'] < 0.01):
            print('Abort learning, dice_coef=', logs['dice_coef'])
            # print('Abort learning, acc=', logs['accuracy'], file=f)

            self.model.stop_training = True
            return

        CR = logs['val_dice_coef']

        print('\n', CR, self.State[0])
        if CR > self.State[0]:
            self.State[0] = CR
            self.State[1] = epoch

            self.BestW = self.model.get_weights()
            self.model.save(rf'{CrPath}/best0.h5')
            print(f'\nСохранено наибольшее значение {CR}')

            np.save(rf'{CrPath}/cr0.npy', self.BestW)
            print('Сохранено текущее значение')
        else:
            np.save(rf'{CrPath}/cr0.npy', self.model.get_weights())
            print('\nСохранено текущее значение')


try:
    custom_objects = {"dice_coef": dice_coef}
    Model0 = load_model(rf'{CrPath}/best.h5', custom_objects=custom_objects)
    # W = np.load(rf'{CrPath}/cr0.npy')
    print('Прочитана модель')
    # Model0.set_weights(W)

    # Model1.load(rf'{CrPath}/best.h5')
    # W = np.load(rf'{CrPath}/cr0.npy')
    # print ('Прочитано W')
    # Model1.set_weights(W)


except:
    print('Ошибка чтения модели')
    Model0 = unetWithMask()
    # Model1 = unetWithMask()
    try:
        np.load(rf'{CrPath}/cr.npy')
        print('Веса загружены')
    except:
        print('Веса не загружены')

print('Ready')
CrCallback0 = THromosomCallback()
# CrCallback1 = THromosomCallback()

N = 0

# while(True):
gc.collect()
history0 = Model0.fit(InputX0, InputY0, epochs=100, batch_size=8, validation_data=(TestX, TestY),
                      callbacks=[CrCallback0])
# history1 = Model1.fit(InputX1, InputY1, epochs=1, batch_size=16, validation_data=(TestX, TestY), callbacks=[CrCallback1])
# print(f'/nЭпоха {N}, лучшие результаты: {CrCallback0.CR}:{CrCallback1.CR}')
# N+= 1