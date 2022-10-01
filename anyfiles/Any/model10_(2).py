# -*- coding: utf-8 -*-
"""Model10 (2).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10_K23PrriPmsj7aRK2nqMm5B90Gx3yAQ

# Поехали
"""

import numpy as np
import random
from tensorflow.keras.models import Model, load_model # Импортируем модели keras: Model
from tensorflow.keras.layers import Input, Rescaling, Conv2DTranspose, concatenate, Activation, SpatialDropout2D, MaxPooling2D, AveragePooling2D, Conv2D, BatchNormalization # Импортируем стандартные слои keras
from tensorflow.keras import backend as K # Импортируем модуль backend keras'а
from tensorflow.keras.optimizers import Adam, Nadam # Импортируем оптимизатор Adam
from tensorflow.keras import utils # Импортируем модуль utils библиотеки tensorflow.keras для получения OHE-представления
from keras import regularizers
from keras.callbacks import Callback
import tensorflow as tf
import os
from tqdm.auto import tqdm
import glob
import cv2
#import albumentations as A
import gc
from tqdm.auto import tqdm
import json
from torch.utils.data import Dataset

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

ImgSz =256

def MakeAug(image, mask, a = 9, verbose = False, p = 1):
    if a == 0:
        aug = A.HorizontalFlip(p=p)
    elif a == 1:
        aug = A.VerticalFlip(p=p)
    elif a == 2:
        aug = A.RandomRotate90(p=p)
    elif a == 3:
        aug = A.RandomRotate90(p=p)
    elif a == 4:
        aug = A.Transpose(p=p)
    elif a == 5:
        aug = A.ElasticTransform(p=p, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
    elif a == 6:
        aug = A.GridDistortion(p=p)
    elif a == 7:
        aug = A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
    elif a == 8:
        original_height, original_width = image.shape[:2]
        aug = A.RandomSizedCrop(min_max_height=(original_height*0.7, original_width*0.7), height=original_height, width=original_width, p=1)
    elif a == 9:
        original_height, original_width = image.shape[:2]
        aug = A.Compose([
            A.VerticalFlip(p=p/2),
            A.HorizontalFlip(p=p / 2),
            A.RandomRotate90(p=p/2),
            A.CLAHE(p=0.8),
            A.RandomGamma(p=0.8)])



    augmented = aug(image=image, mask=mask)

    image_h_flipped = augmented['image']
    mask_h_flipped = augmented['mask']

    return image_h_flipped, mask_h_flipped

class EyeDataset(Dataset):
    """
    Класс датасета, организующий загрузку и получение изображений и соответствующих разметок
    """

    def __init__(self, data_folder: str, transform=None, ImgReadMode=cv2.IMREAD_COLOR):
        self.class_ids = {"vessel": 1}

        self.data_folder = data_folder
        self.transform = transform
        self._image_files = glob.glob(f"{CrPath}{data_folder}/f*.png")
        
        EyeDataset.ImgReadMode = ImgReadMode  # знаю, что так использовать стат перем плохая идея

        self.CrImgFile = ''

    @staticmethod
    def read_image(path: str) -> np.ndarray:
        image = cv2.imread(str(path), EyeDataset.ImgReadMode)

        if EyeDataset.ImgReadMode != cv2.IMREAD_GRAYSCALE:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.ubyte)

        #if image.shape[0] * image.shape[0] != 256*256:
        #    print('Ошибка размера Img') 
        #    print(path, image.shape)

        return image

    @staticmethod
    def parse_polygon(coordinates: dict, image_size: tuple) -> np.ndarray:
        mask = np.zeros(image_size, dtype=np.ubyte)
        if len(coordinates) == 1:
            points = [np.int32(coordinates)]
            cv2.fillPoly(mask, points, 1)
        else:
            for polygon in coordinates:
                points = [np.int32([polygon])]
                cv2.fillPoly(mask, points, 1)
        return mask

    @staticmethod
    def parse_mask(shape: dict, image_size: tuple) -> np.ndarray:
        """
        Метод для парсинга фигур из geojson файла
        """
        mask = np.zeros(image_size, dtype=np.ubyte)
        coordinates = shape['coordinates']
        if shape['type'] == 'MultiPolygon':
            for polygon in coordinates:
                mask += EyeDataset.parse_polygon(polygon, image_size)
        else:
            mask += EyeDataset.parse_polygon(coordinates, image_size)

        return mask

    def read_layout(self, path: str, image_size: tuple) -> np.ndarray:
        """
        Метод для чтения geojson разметки и перевода в numpy маску
        """
        with open(path, 'r', encoding='cp1251') as f:  # some files contain cyrillic letters, thus cp1251
            json_contents = json.load(f)

        num_channels = 1 + max(self.class_ids.values())
        mask_channels = [np.zeros(image_size, dtype=np.ubyte) for _ in range(num_channels)]
        mask = np.zeros(image_size, dtype=np.ubyte)

        if type(json_contents) == dict and json_contents['type'] == 'FeatureCollection':
            features = json_contents['features']
        elif type(json_contents) == list:
            features = json_contents
        else:
            features = [json_contents]

        for shape in features:
            channel_id = self.class_ids["vessel"]
            mask = self.parse_mask(shape['geometry'], image_size)
            mask_channels[channel_id] = np.maximum(mask_channels[channel_id], mask)

        mask_channels[0] = 1 - np.max(mask_channels[1:], axis=0)

        return np.stack(mask_channels, axis=-1)

    def __getitem__(self, idx: int) -> dict:
        # Достаём имя файла по индексу
        image_path = self._image_files[idx]

        self.CrImgFile = image_path

        # Получаем соответствующий файл разметки
        m_path = image_path.replace(r"/f", r"/m")

        image = self.read_image(image_path)
        
        mask = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        #print(m_path)
        #self.read_layout(json_path, image.shape[:2])
        #if mask.shape[0] * mask.shape[0] != 256*256:
        #    print('Ошибка размера mask') 
        #    print(m_path, mask.shape) 
        sample = {'image': image,
                  'mask': mask/255}
        
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    def __len__(self):
        return len(self._image_files)

    # Метод для проверки состояния датасета
    def make_report(self):
        reports = []
        if (not self.data_folder):
            reports.append("Путь к датасету не указан")
        if (len(self._image_files) == 0):
            reports.append("Изображения для распознавания не найдены")
        else:
            reports.append(f"Найдено {len(self._image_files)} изображений")
        cnt_images_without_masks = sum(
            [1 - len(glob.glob(filepath.replace("png", "geojson"))) for filepath in self._image_files])
        if cnt_images_without_masks > 0:
            reports.append(f"Найдено {cnt_images_without_masks} изображений без разметки")
        else:
            reports.append(f"Для всех изображений есть файл разметки")
        return reports

def unetWithMask(num_classes=2, input_shape=(ImgSz, ImgSz, 1)):
    img_input = Input(input_shape)  # Создаем входной слой с размерностью input_shape

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(img_input)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_1_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_1_out

    block_1_out_mask = Conv2D(64, (1, 1), padding='same', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(
        block_1_out)  # Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_1_out_mask

    x = MaxPooling2D()(block_1_out)  # Добавляем слой MaxPooling2D

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_2_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_2_out

    block_2_out_mask = Conv2D(128, (1, 1), padding='same', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(
        block_2_out)  # Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_2_out_mask

    x = MaxPooling2D()(block_2_out)  # Добавляем слой MaxPooling2D

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_3_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_3_out

    block_3_out_mask = Conv2D(256, (1, 1), padding='same', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(
        block_3_out)  # Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_3_out_mask

    x = MaxPooling2D()(block_3_out)  # Добавляем слой MaxPooling2D

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    block_4_out = Activation('relu')(x)  # Добавляем слой Activation и запоминаем в переменной block_4_out

    block_4_out_mask = Conv2D(512, (1, 1), padding='same', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(
        block_4_out)  # Добавляем Conv2D-маску к текущему слою и запоминаем в переменную block_4_out_mask

    x = MaxPooling2D()(block_4_out)  # Добавляем слой MaxPooling2D

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 1
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(
        x)  # Добавляем слой Conv2DTranspose с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_4_out,
                     block_4_out_mask])  # Объединем текущий слой со слоем block_4_out и слоем-маской block_4_out_mask
    x = Conv2D(512, (3, 3), padding='same', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем слой Conv2D с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем слой Conv2D с 512 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 2
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(
        x)  # Добавляем слой Conv2DTranspose с 256 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_3_out,
                     block_3_out_mask])  # Объединем текущий слой со слоем block_3_out и слоем-маской block_3_out_mask
    x = Conv2D(256, (3, 3), padding='same', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем слой Conv2D с 256 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем слой Conv2D с 256 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(
        x)  # Добавляем слой Conv2DTranspose с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_2_out,
                     block_2_out_mask])  # Объединем текущий слой со слоем block_2_out и слоем-маской block_2_out_mask
    x = Conv2D(128, (3, 3), padding='same', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    # UP 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)  # Добавляем слой Conv2DTranspose с 64 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = concatenate([x, block_1_out,
                     block_1_out_mask])  # Объединем текущий слой со слоем block_1_out и слоем-маской block_1_out_mask
    x = Conv2D(64, (3, 3), padding='same', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same', activity_regularizer = regularizers.L1(0.002), kernel_regularizer = regularizers.L2(0.002))(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)  # Добавляем слой BatchNormalization
    x = Activation('relu')(x)  # Добавляем слой Activation

    x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(
        x)  # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов

    model = Model(img_input, x)  # Создаем модель с входом 'img_input' и выходом 'x'

    # Компилируем модель
    model.compile(optimizer=Nadam(learning_rate=0.005),
                  loss='categorical_crossentropy')

    return model  # Возвращаем сформированную модель


def dice_coef_np(y_true, y_pred):
    return (2. * np.sum(y_true * y_pred) + 1.) / (np.sum(y_true) + np.sum(y_pred) + 1.)

X = []
Y = []
InBatch = 0
Au = 0

def Generate(dataset, BatchSz = 16, AuSteps = 89, Test = False, ImgSz = ImgSz):
    InBatch = BatchSz
    

    X = []
    Y = []

    while(True):
        random.seed(11)
        for Au in range(AuSteps):
            for sample in dataset:

                # if Step < FileIndex*50: # аккуратненько чтоб не вникать в тонкости устройства dataset пропускаем уже сделанные позиции (использую при обрыве выполнения)
                #    Step+= 1
                # continue

                m = sample["mask"][:, :]
                Img = sample['image']

                Sz = Img.shape[:2]

                #Img = cv2.resize(Img, (ImgSz, ImgSz))
                Mask = m #cv2.resize(m, (ImgSz, ImgSz))

                #if Au > 0 or Test:
                    #Img, Mask = MakeAug(Img, Mask, verbose= False)

                if InBatch > 0:
                    Mask = Mask.reshape((1, ImgSz, ImgSz, 1))
                    Mask1 = 1 - Mask

                    Mask = np.concatenate([Mask, Mask1], 3)
                    X.append(Img.reshape((1, ImgSz, ImgSz, 1)))
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

'''
try:
    model = load_model(rf'{CrPath}model10.h5')
    model.load_weights(rf'{CrPath}cr.npy')
    print('Прочитаны веса')

except:
    try:
        print(rf'{CrPath}model10.h5')
        model = load_model(rf'{CrPath}model10.h5')
        print('Прочитана модель')
    except:
        model = unetWithMask()
        print('Модель создана со случайными весами')
'''
'''
model = load_model(rf'{CrPath}model10.h5')
W2 = model.get_weights()
#W2 = np.load('e:/musor/nw10.npy', allow_pickle=True)
model = unetWithMask()
model.load_weights(rf'{CrPath}W10.h5')
W1 = model.get_weights()

for i in range(len(W2)):
    A0 = W2[i]
    A1 = W1[i]

    print(i, (A0 != A1).sum())

    a = 0
'''
model = unetWithMask()

try:
    model.load_weights(rf'{CrPath}W10.h5')
    print('Прочитаны веса')
except:
    print('Не удалось загрузить веса')

#model = load_model(rf'{CrPath}W10.h5')

Best = 0
Index = 0
dataset = EyeDataset("trainfull", ImgReadMode=cv2.IMREAD_GRAYSCALE)

G = Generate(dataset)


'''
Test = Generate(dataset, Test = True)
print('Поехали')

#TestX = [0]*64
#TestY = [0]*64

#X, Y = next(G)

print('Генерируем Test набор')
index = 0

TestX = [None]*128
TestY = [None]*128 

try:
    TestX = np.load('TestX.npy')
    TestY = np.load('TestY.npy')
except:
    for index in tqdm(range(128)):
        X, Y = next(Test)

        TestX[index] = X
        TestY[index] = Y

    try:
        TestX = np.concatenate(TestX)
        TestY = np.concatenate(TestY)
    except:
        print('')

    np.save(fr'{CrPath}TestX1.npy', TestX)
    np.save(fr'{CrPath}TestY1.npy', TestY)
'''
print('Читаем тестовый набор')
TestX = np.load(fr'{CrPath}TestX1.npy')
TestY = np.load(fr'{CrPath}TestY1.npy')
print('Поехали')

while True:
    #X, Y = next(G)
    history = model.fit(G, epochs=1, steps_per_epoch=1, batch_size=16, verbose=1, use_multiprocessing=True)
    
    model.save(rf'{CrPath}model10.h5')
    model.save_weights(rf'{CrPath}W10.h5')
    gc.collect()

    Res = model.predict(TestX[:512], verbose=True)[:, :, :, 0:1]
    er = dice_coef_np(Res, TestY[:512])   

    if er > Best:
        Best = er
        model.save(rf'{CrPath}Best10.h5')
        model.save_weights(rf'{CrPath}BestW10.h5')
        print('Сохранена лучшая модель. ', Index, er, Best)
    else:
        print(Index, er, Best)

    Index += 1
    #np.save(rf'{CrPath}Res.npy', Res)