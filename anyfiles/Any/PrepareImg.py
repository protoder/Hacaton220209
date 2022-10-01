import cv2
import numpy as np
import pandas as pd
from glob import glob
from tqdm.auto import tqdm
from sklearn.metrics import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from sklearn.model_selection import train_test_split
import os
from sklearn.neighbors import NearestNeighbors
import random
import json
from torch.utils.data import Dataset
import glob

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


class EyeDataset(Dataset):
    """
    Класс датасета, организующий загрузку и получение изображений и соответствующих разметок
    """

    def __init__(self, data_folder: str, transform=None, ImgReadMode=cv2.IMREAD_COLOR):
        self.class_ids = {"vessel": 1}

        self.data_folder = data_folder
        self.transform = transform
        self._image_files = glob.glob(f"{CrPath}{data_folder}/*.png")
        EyeDataset.ImgReadMode = ImgReadMode  # знаю, что так использовать стат перем плохая идея

        self.CrImgFile = ''

    @staticmethod
    def read_image(path: str) -> np.ndarray:

        image = cv2.imread(str(path), EyeDataset.ImgReadMode)

        if EyeDataset.ImgReadMode != cv2.IMREAD_GRAYSCALE:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.ubyte)
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

        self.CrFile = image_path
        self.CrImgFile = image_path

        # Получаем соответствующий файл разметки
        json_path = image_path.replace("png", "geojson")

        image = self.read_image(image_path)

        mask = self.read_layout(json_path, image.shape[:2])

        sample = {'image': image,
                  'mask': mask}

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


def GenerateTrainSmall():
    dataset = EyeDataset("train", ImgReadMode=cv2.IMREAD_GRAYSCALE)

    Res = np.array([], dtype=np.ubyte)
    Index = 0
    FileIndex = 0
    Step = 0
    for sample in tqdm(dataset):

        # if Step < FileIndex*50: # аккуратненько чтоб не вникать в тонкости устройства dataset пропускаем уже сделанные позиции (использую при обрыве выполнения)
        #    Step+= 1
        #    continue

        m = sample["mask"][:, :, 1]
        Img = sample['image']

        Shape = Img.shape

        if Shape[0] != 1232 or Shape[1] != 1624:
            Img = cv2.resize(Img, (1624, 1232))
            m = cv2.resize(m, (1624, 1232))

            # cv2.imshow('', Img * (1 - m))
            # cv2.imshow('', Img)
            # cv2.waitKey(0)

        # continue

        Res = np.append(Res, Img[Left:Left + 120, Top: Top + 120])
        Res = np.append(Res, m[Left:Left + 120, Top: Top + 120])

        Index += 1
        if Index == 50 or Index == len(dataset):
            Res = np.array(Res).reshape((Index, 12, 16, 2, 120, 120))

            np.save(f'numpy/train{FileIndex}', Res)

            FileIndex += 1
            # print(f'Сформирован файл /numpy/train{Index}.np')

            Index = 0

            Res = np.array([], dtype=np.ubyte)

    Res = np.array(Res).reshape((Index, 12, 16, 2, 120, 120))

    np.save(f'numpy/train{FileIndex}', Res)

    # ResImg = np.concatenate((Img, m), axis=0)
    ''' 
    cv2.imshow('', Img)
    #cv2.imshow('', Img)
    cv2.waitKey(1500)

    cv2.imshow('', m)
    cv2.waitKey(1500)

    cv2.imshow('', Img * (m))
    # cv2.imshow('', Img)
    cv2.waitKey(0)
    '''


'''
def GenerateLargeTrainl():
    dataset = EyeDataset("train", ImgReadMode = cv2.IMREAD_GRAYSCALE)

    Res = np.array([], dtype = np.ubyte)
    Index = 0
    FileIndex = 2
    Step = 0
    for sample in tqdm(dataset):

        if Step < FileIndex*50: # аккуратненько чтоб не вникать в тонкости устройства dataset пропускаем уже сделанные позиции (использую при обрыве выполнения)
            Step+= 1
            continue

        m = sample["mask"][:, :, 1]
        Img = sample['image']

        Shape = Img.shape

        if Shape[0] != 1232 or Shape[1] != 1624:
            Img = cv2.resize(Img, (1624, 1232) )
            m  = cv2.resize(m, (1624, 1232) )

            #cv2.imshow('', Img * (1 - m))
            # cv2.imshow('', Img)
            #cv2.waitKey(0)

        #continue

        for x in range(12):


        Index += 1
        if Index == 50 or Index == len(dataset):
            Res = np.array(Res).reshape( (Index , 12, 16, 2, 120, 120))

            np.save(f'numpy/train{FileIndex}', Res)

            FileIndex+= 1
            #print(f'Сформирован файл /numpy/train{Index}.np')

            Index = 0

            Res = np.array([], dtype = np.ubyte)

    Res = np.array(Res).reshape((Index, 12, 16, 2, 120, 120))

    np.save(f'numpy/train{FileIndex}', Res)

    # ResImg = np.concatenate((Img, m), axis=0)

    cv2.imshow('', Img)
    #cv2.imshow('', Img)
    cv2.waitKey(1500)

    cv2.imshow('', m)
    cv2.waitKey(1500)

    cv2.imshow('', Img * (m))
    # cv2.imshow('', Img)
    cv2.waitKey(0)
    '''


# Slides - в формате [Ansambles, 128, 128, 1]
def RestoreSmallImg(Slides):
    SumRes = np.zeros((1232, 1624), np.float32)

    for ASlides in Slides:
        Res = np.zeros((1232, 1624), np.byte)
        for x, SlideX in enumerate(ASlides):
            if x == 11:
                Left = 1232 - 128
            else:
                Left = 103 * x

            for y, Slide in enumerate(SlideX):

                if y == 15:
                    Top = 1624 - 128
                else:
                    Top = 102 * y

                Res[Left:Left + 128, Top: Top + 128] |= Slide

        SumRes += Res

    Res = (SumRes / len(Slides)) >= 0.5

    return Res


def GenerateTrainColored():
    dataset = EyeDataset("train", ImgReadMode=cv2.IMREAD_COLOR)
    Ress = []
    Res = np.array([], dtype=np.ubyte)
    Index = 0
    FileIndex = 2
    Step = 0
    for sample in tqdm(dataset):

        if Step == FileIndex * 50:  # аккуратненько чтоб не вникать в тонкости устройства dataset пропускаем уже сделанные позиции (использую при обрыве выполнения)
            Step += 1
            # continue

        m = sample["mask"][:, :, 1]
        Img = sample['image']

        Shape = Img.shape

        if Shape[0] != 1232 or Shape[1] != 1624:
            Img = cv2.resize(Img, (1624, 1232))
            m = cv2.resize(m, (1624, 1232))

            # cv2.imshow('', Img * (1 - m))
            # cv2.imshow('', Img)
            # cv2.waitKey(0)

        # continue

        for x in range(12):
            if x == 11:
                Left = 1232 - 128
            else:
                Left = 103 * x

            for y in range(16):
                if y == 15:
                    Top = 1624 - 128
                else:
                    Top = 102 * y

                Y = m[Left:Left + 128, Top: Top + 128]
                if Y.sum() > 0:
                    Res = np.append(Res, Img[Left:Left + 128, Top: Top + 128])
                    Res = np.append(Res, Y)

        Index += 1

        if Index == 50 or Index == len(dataset):
            if len(Ress == 2):
                break
            L = len(Res) // (2 * 128 * 128)
            if L * (2 * 128 * 128) != len(Res):
                print('Error!')
            Res = np.array(Res).reshape((L, 2, 128, 128))

            # np.save(f'numpy/train{FileIndex}', Res)

            Ress.append(Res)

            FileIndex += 1
            # print(f'Сформирован файл /numpy/train{Index}.np')

            Index = 0

            Res = np.array([], dtype=np.ubyte)

    L = len(Res) // (2 * 128 * 128)
    if L * (2 * 128 * 128) != len(Res):
        print('Error!')
    Res = np.array(Res).reshape((L, 2, 128, 128))

    Ress.append(Res)

    RN = np.concatenate(Ress)

    np.save(f'numpy/train{FileIndex}', RN)

    # ResImg = np.concatenate((Img, m), axis=0)
    ''' 
    cv2.imshow('', Img)
    #cv2.imshow('', Img)
    cv2.waitKey(1500)

    cv2.imshow('', m)
    cv2.waitKey(1500)

    cv2.imshow('', Img * (m))
    # cv2.imshow('', Img)
    cv2.waitKey(0)
    '''


def GenerateTrainl():
    dataset = EyeDataset("train", ImgReadMode=cv2.IMREAD_GRAYSCALE)

    Ress = []
    Res = np.array([], dtype=np.ubyte)
    Index = 0
    FileIndex = 2
    Step = 0
    for sample in tqdm(dataset):

        # if Step < FileIndex*50: # аккуратненько чтоб не вникать в тонкости устройства dataset пропускаем уже сделанные позиции (использую при обрыве выполнения)
        #    Step+= 1
        # continue

        m = sample["mask"][:, :, 1]
        Img = sample['image']

        Shape = Img.shape

        if Shape[0] != 1232 or Shape[1] != 1624:
            Img = cv2.resize(Img, (1624, 1232))
            m = cv2.resize(m, (1624, 1232))

            # cv2.imshow('', Img * (1 - m))
            # cv2.imshow('', Img)
            # cv2.waitKey(0)

        # continue

        for x in range(12):
            if x == 11:
                Left = 1232 - 128
            else:
                Left = 103 * x

            for y in range(16):
                if y == 15:
                    Top = 1624 - 128
                else:
                    Top = 102 * y

                Y = m[Left:Left + 128, Top: Top + 128]
                Res = np.append(Res, Img[Left:Left + 128, Top: Top + 128])
                Res = np.append(Res, Y)

        Index += 1

        if Index == 25 or Index == len(dataset):

            L = len(Res) // (2 * 128 * 128)
            if L * (2 * 128 * 128) != len(Res):
                print('Error!')
            Res = np.array(Res).reshape((L, 2, 128, 128))

            # np.save(f'numpy/train{FileIndex}', Res)

            Ress.append(Res)

            FileIndex += 1
            # print(f'Сформирован файл /numpy/train{Index}.np')

            Index = 0

            Res = np.array([], dtype=np.ubyte)

    L = len(Res) // (2 * 128 * 128)
    if L * (2 * 128 * 128) != len(Res):
        print('Error!')
    Res = np.array(Res).reshape((L, 2, 128, 128))

    Ress.append(Res)

    RN = np.concatenate(Ress)

    np.save(f'{CrPath}numpy/train2', RN)

    # ResImg = np.concatenate((Img, m), axis=0)
    ''' 
    cv2.imshow('', Img)
    #cv2.imshow('', Img)
    cv2.waitKey(1500)

    cv2.imshow('', m)
    cv2.waitKey(1500)

    cv2.imshow('', Img * (m))
    # cv2.imshow('', Img)
    cv2.waitKey(0)
    '''

def GenerateTrain256():
    dataset = EyeDataset("train", ImgReadMode=cv2.IMREAD_GRAYSCALE)

    Ress = []
    Res = np.array([], dtype=np.ubyte)
    Index = 0
    FileIndex = 2
    Step = 0
    for sample in tqdm(dataset):

        # if Step < FileIndex*50: # аккуратненько чтоб не вникать в тонкости устройства dataset пропускаем уже сделанные позиции (использую при обрыве выполнения)
        #    Step+= 1
        # continue

        m = sample["mask"][:, :, 1]
        Img = sample['image']

        Shape = Img.shape

        if Shape[0] != 1232 or Shape[1] != 1624:
            Img = cv2.resize(Img, (1624, 1232))
            m = cv2.resize(m, (1624, 1232))

            # cv2.imshow('', Img * (1 - m))
            # cv2.imshow('', Img)
            # cv2.waitKey(0)

        # continue

        for x in range(12):
            if x == 11:
                Left = 1232 - 128
            else:
                Left = 103 * x

            for y in range(16):
                if y == 15:
                    Top = 1624 - 128
                else:
                    Top = 102 * y

                Y = m[Left:Left + 128, Top: Top + 128]
                Res = np.append(Res, Img[Left:Left + 128, Top: Top + 128])
                Res = np.append(Res, Y)

        Index += 1

        if Index == 25 or Index == len(dataset):

            L = len(Res) // (2 * 128 * 128)
            if L * (2 * 128 * 128) != len(Res):
                print('Error!')
            Res = np.array(Res).reshape((L, 2, 128, 128))

            # np.save(f'numpy/train{FileIndex}', Res)

            Ress.append(Res)

            FileIndex += 1
            # print(f'Сформирован файл /numpy/train{Index}.np')

            Index = 0

            Res = np.array([], dtype=np.ubyte)

    L = len(Res) // (2 * 128 * 128)
    if L * (2 * 128 * 128) != len(Res):
        print('Error!')
    Res = np.array(Res).reshape((L, 2, 128, 128))

    Ress.append(Res)

    RN = np.concatenate(Ress)

    np.save(f'{CrPath}numpy/train2', RN)

    # ResImg = np.concatenate((Img, m), axis=0)
    ''' 
    cv2.imshow('', Img)
    #cv2.imshow('', Img)
    cv2.waitKey(1500)

    cv2.imshow('', m)
    cv2.waitKey(1500)

    cv2.imshow('', Img * (m))
    # cv2.imshow('', Img)
    cv2.waitKey(0)
    '''

def GenerateOneFile():
    for i in range(8):
        File = np.load(fr'numpy/train{i}.npy')

        if i == 0:
            Train = File
        else:
            Train = np.concatenate((Train, File))
    np.save(r'numpy/train.npy', Train)


def DemonstrateImgSmall():
    Data = np.load(r'numpy/train.npy')
    for N in range(387):
        for x in range(12):
            for y in range(16):
                Img = Data[N, x, y, 0, :, :].reshape(120, 120, 1)
                cv2.imwrite(fr'numpy/train{N}_{x}_{y}.png', Img)
                Img = Data[N, x, y, 1, :, :].reshape(120, 120, 1) * 255
                cv2.imwrite(fr'numpy/test{N}_{x}_{y}.png', Img)

from HacatonMain import RestoreSmallImg

def DemonstrateMasks():
    Data = np.load(r'numpy/train without shufl.npy')
    Data = Data.reshape( (len(Data) // (12*16), 12, 16, 2, 128, 128, 1))
    files = glob.glob(f"{CrPath}train/*.png")
    for N, Img in enumerate(Data[:,:,:,1,:,:,:]):
        Img = Img.reshape(1, 12, 16, 128, 128, 1)
        Res = RestoreSmallImg(Img) * 255

        #basename = os.path.basename(files[N])
        cv2.imwrite(fr'Res/m{N}.png', Res)

if __name__ == '__main__':
    import tensorflow as tf

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    DemonstrateMasks()
    GenerateTrainl()