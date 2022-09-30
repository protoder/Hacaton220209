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
import tensorflow as tf
import os
from tqdm.auto import tqdm
import glob
import cv2
import json
from torch.utils.data import Dataset

GrayScaled = True

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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

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
    def parse_polygon2(coordinates: dict, image_size: tuple) -> np.ndarray:
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
    def parse_polygon(coordinates: dict, image_size: tuple) -> np.ndarray:
        mask = np.zeros(image_size, dtype=np.ubyte)

        if len(coordinates) == 1:
            points = [np.int32(coordinates)]
            cv2.fillPoly(mask, points, 1)
        else:
            points = [np.int32([coordinates[0]])]
            cv2.fillPoly(mask, points, 1)

            for polygon in coordinates[1:]:
                points = [np.int32([polygon])]
                cv2.fillPoly(mask, points, 0)

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


def read_image(path: str, GrayScaled) -> np.ndarray:
    if GrayScaled:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(imageCl, cv2.COLOR_BGR2RGB)

    image = np.array(image, dtype=np.ubyte)  # .reshape

    return image


def GenerateTrain(path, GrayScaled=True):
    _image_files = glob.glob(f"{path}/*.png")

    Ress = []
    Res = np.array([], dtype=np.ubyte)
    Index = 0
    HallIndex = 0
    FileIndex = 2
    Step = 0

    for f in tqdm(_image_files):
        Img = read_image(f, GrayScaled)

        # if Step < FileIndex*50: # аккуратненько чтоб не вникать в тонкости устройства dataset пропускаем уже сделанные позиции (использую при обрыве выполнения)
        Step += 1
        # continue

        Shape = Img.shape

        if Shape[0] != 1232 or Shape[1] != 1624:
            Img = cv2.resize(Img, (1624, 1232), cv2.INTER_AREA)


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

                Res = np.append(Res, Img[Left:Left + 128, Top: Top + 128])

        Index += 1

        if Index == 18 or HallIndex == len(_image_files) - 1:
            L = len(Res) // (128 * 128)
            if L * (128 * 128) != len(Res):
                print('Error!')
            Res = np.array(Res).reshape((L, 128, 128, 1))

            # np.save(f'numpy/train{FileIndex}', Res)

            Ress.append(Res)

            FileIndex += 1
            # print(f'Сформирован файл /numpy/train{Index}.np')

            Index = 0

            Res = np.array([], dtype=np.ubyte)

        HallIndex += 1

    return np.concatenate(Ress), _image_files


def RestoreSmallImg(Slides):
    SumRes = np.zeros((1232, 1624), np.float32)

    for ASlides in Slides:
        Res = np.zeros((1232, 1624), np.bool)
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

                Res[Left:Left + 128, Top: Top + 128] |= ((Slide > 0.4).reshape((128, 128)))

        SumRes += Res

    Res = (1 - (np.array(SumRes / len(Slides)) >= 0.5)).reshape((1232, 1624, 1))

    return Res


# X в формате [N, 128, 128, 1]
def PredictList(X, GrayScaled=True, Model=None):
    Test = np.array(X, np.float32)

    ResTest = model.predict(Test, verbose=True)[:, :, :, 0:1]

    Shape = list(ResTest.shape)

    ResTest = ResTest.reshape((Shape[0] // (12 * 16), 12, 16, Shape[-3], Shape[-2], Shape[-1]))

    for Img, F in zip(ResTest, Files):
        Res = RestoreSmallImg(Img.reshape((1, 12, 16, 128, 128)))

        Res = np.array(Res, dtype=np.byte)
        basename = os.path.basename(F)
        cv2.imwrite(fr'dRAW_{ResPath}{os.path.splitext(basename)[0]}.png', Res)
        cv2.imwrite(fr'{ResPath}{os.path.splitext(basename)[0]}.png', Res)


def Predict(path, ResPath, ModelPath, GrayScaled=True, Model=None):
    try:
        Test = np.load(fr'{path}test.npy')
        # np.save(fr'{path}test.npy', Test)
        Files = []
        with open(rf"{CrPath}Files.lst", 'r') as filehandle:
            for line in filehandle:
                # удалим заключительный символ перехода строки
                currentPlace = line[:-1]

                # добавим элемент в конец списка
                Files.append(currentPlace)
    except:
        Test, Files = GenerateTrain(path, GrayScaled)
        # Здесь формат [N, 128, 128, 1]
        np.save(fr'{path}test.npy', Test)
        print('Test сохранен')
        with open(rf"{CrPath}Files.lst", 'w') as filehandle:
            for listitem in Files:
                filehandle.write('%s\n' % listitem)
        print('Файлы сохранены')

    if Model == None:
        files = glob.glob(f"{ModelPath}*.h5")
        for M in files:
            model = load_model(M)
    else:
        model = Model
    # Test = Test.reshape(Test.shape[0] * Test.shape[1], 128, 128, 1 if GrayScaled else 3)
    # Test = np.array(Test, np.float32)

    StartPos = 0
    EndPos = 30 * (12 * 16)

    StartFilePos = 0
    EndFilePos = 30
    while EndFilePos < len(Test):

        ResTest = model.predict(Test[StartPos:EndPos], verbose=True)[:, :, :, 0:1]

        Shape = list(ResTest.shape)

        ResTest = ResTest.reshape((Shape[0] // (12 * 16), 12, 16, Shape[-3], Shape[-2], Shape[-1]))

        Index = 0

        for Img, F in zip(ResTest, Files[StartFilePos:EndFilePos]):
            Res = RestoreSmallImg(Img.reshape((1, 12, 16, 128, 128)))

            Res = np.array(Res, dtype=np.byte)
            basename = os.path.basename(F)
            FileName = os.path.splitext(basename)[0]
            # cv2.imwrite(fr'{ResPath}dRAW_{os.path.splitext(basename)[0]}.png', Res*255)
            # cv2.imwrite(fr'{ResPath}{os.path.splitext(basename)[0]}.png', Res)
            cv2.imwrite(fr'{ResPath}dRAW_{FileName}.png', Res * 255)
            cv2.imwrite(fr'{ResPath}{FileName}.png', Res)

            Index += 1

        StartFilePos = EndFilePos
        StartPos = EndPos

        EndFilePos += 30
        EndPos += 30 * (12 * 16)


def GenerateTestImg(CrPath, ResPath, Files):
    Data = np.load(rf'{CrPath}numpy/train.npy')

    InputY = Data[:, 1]

    InputY.reshape((len(InputY) // (12 * 16), 12, 16, 128, 128))

    for Img, F in zip(ResTest, Files):
        Res = RestoreSmallImg(Img.reshape((1, 12, 16, 128, 128)))

        Res = np.array(Res, dtype=np.byte)
        basename = os.path.basename(F)
        cv2.imwrite(fr'mDRAW_{ResPath}{os.path.splitext(basename)[0]}.png', Res)
        cv2.imwrite(fr'm_{ResPath}{os.path.splitext(basename)[0]}.png', Res)

def ClearFields(Img, Step = 20, Verbose = True, Inverse = True, FileName = 'Тест'):
    Shape = Img.shape

    Img1 = np.array(Img//255, dtype=np.uint8) # теперь линии соответствует 1
    #Img1 = np.array(Img * 255, dtype=np.uint8)  # теперь линии соответствует 1
    #return Img1

    if Inverse:
        Img1 = 1 - Img1

    # Вертикальная очистка
    YHalf = Shape[0] // 2

    Vert = np.zeros_like(Img, dtype=np.uint8)

    for i in range(Step):
        Vert[:YHalf, :] += Img1[i:i+YHalf, :]

        if i == 0:
            Last = None
        else:
            Last = -i

        Vert[YHalf:, :] += Img1[YHalf-i:Last, :]

    YMask = Vert < Step // 4 * 3 # 0 там, где надо обнулить
    MaskCopy = YMask.copy()
    for i in range(Step-1):
        YMask[i+1:i + 1 + YHalf, :] *= MaskCopy[:YHalf, :]

        YMask[YHalf - i - 1:-1-i, :]  *= MaskCopy[YHalf:, :]

    # Горизонтальная очистка
    XHalf = Shape[1] // 2
    Hor = np.zeros_like(Img, dtype=np.uint8)

    for i in range(Step):
        Hor[:,:XHalf] += Img1[:, i:i + XHalf]

        if i == 0:
            Last = None
        else:
            Last = -i

        Hor[:, XHalf:] += Img1[:,XHalf - i:Last]

    XMask = Hor < Step // 4 * 3  # 0 там, где надо обнулить

    MaskCopy = XMask.copy()
    for i in range(Step-1):
        XMask[:, i+1:i + 1 + XHalf] *= MaskCopy[:, :XHalf]

        XMask[:, XHalf - i - 1:-1-i]  *= MaskCopy[:, XHalf:]



    Img1 *= XMask * YMask

    if Verbose:
        Hori = np.concatenate((Img, 255*Img1), axis=1)

        # concatenate image Vertically
        cv2.imshow(FileName, Hori)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 255*Img1

def PrepareTest(Img, SegMask = None, Step = 40, Verbose = True, Inverse = True, FileName = 'Тест', OnlyColor = False):
    Shape = Img.shape
    Img0 = Img.copy()
    #Img1 = ColorImg[:,:,0]
    #cv2.imwrite("E:/musor/debug2.png", Img)
    X0 = 0
    X1 = 0
    Y0 = 0
    Y1 = 0

    Img1 = np.array(Img<30, dtype=np.uint8) # теперь линии соответствует 1
    #Img1 = np.array(Img * 255, dtype=np.uint8)  # теперь линии соответствует 1
    #return Img1

    # Вертикальная очистка
    YHalf = Shape[0] // 2

    Vert = np.zeros_like(Img, dtype=np.uint8)

    for i in range(Step):
        Vert[:YHalf, :] += Img1[i:i+YHalf, :]

        if i == 0:
            Last = None
        else:
            Last = -i

        Vert[YHalf:, :] += Img1[YHalf-i:Last, :]

    YMask = Vert < Step // 4 * 3 # 0 там, где надо обнулить
    MaskCopy = YMask.copy()
    for i in range(Step-1):
        YMask[i+1:i + 1 + YHalf, :] *= MaskCopy[:YHalf, :]

        YMask[YHalf - i - 1:-1-i, :]  *= MaskCopy[YHalf:, :]

    # Горизонтальная очистка
    XHalf = Shape[1] // 2
    Hor = np.zeros_like(Img1, dtype=np.uint8)

    for i in range(Step):
        Hor[:,:XHalf] += Img1[:, i:i + XHalf]

        if i == 0:
            Last = None
        else:
            Last = -i

        Hor[:, XHalf:] += Img1[:,XHalf - i:Last]

    XMask = Hor < Step // 4 * 3  # 0 там, где надо обнулить

    MaskCopy = XMask.copy()
    for i in range(Step-1):
        XMask[:, i+1:i + 1 + XHalf] *= MaskCopy[:, :XHalf]

        XMask[:, XHalf - i - 1:-1-i]  *= MaskCopy[:, XHalf:]


    SumMask = XMask * YMask
    SumMask[Img < 20] = 0




    non_zero_a = np.nonzero(SumMask)
    X0 = non_zero_a[0].min()
    X1 = non_zero_a[0].max()
    Y0 = non_zero_a[1].min()
    Y1 = non_zero_a[1].max()




    Avg = 115 #(Img * SumMask).sum() / (SumMask.sum() + 0.00001)

    Mask = np.full_like(XMask, round(Avg), dtype=np.uint8)

    Img = np.where(SumMask == 0, Mask, Img)



    Img = Img[X0:X1+1, Y0:Y1+1]

    if SegMask is not None:
        SegMask = SegMask[X0:X1+1, Y0:Y1+1]



    '''
    XAdd0 = Img[2:, :] - Img[:-2, :]
    XAdd1 = Img[20:, :] - Img[:-20, :]

    Img = np.array(255*((XAdd0[1:-17,:] - XAdd1) > 15),  dtype=np.uint8)
    #XAdd += XAdd.min()
    
    XAdd *= 255 // XAdd.max()

    #Sum = 10*XAdd + Img[:-2, :]

    #Sum[Sum < 0] = 0
    #Sum[Sum > 255] = 255

    Img = np.array(XAdd, dtype=np.uint8)
    
    Add = 2*(Img - Avg)
    Sum = Img + Add

    Sum[Sum < 0] = Avg
    Sum[Sum > 255] = Avg

    Img = np.array(Sum, dtype = np.uint8)

    #Img =

    Img[Img < 20] = Avg
    Img[Img > 200] = Avg
    '''


    if Verbose:
        I0 = cv2.resize(Img, (800, 800))
        I1 = cv2.resize(Img0, (800, 800))

        if SegMask is None:
            Hori = np.concatenate((I0, I1), axis=1)
        else:
            m = cv2.resize(SegMask, (800, 800))
            Hori = np.concatenate((I0, I1, m * 255), axis=1)

        # concatenate image Vertically
        #cv2.imshow(FileName, I1)
        #cv2.imshow(FileName, I1)
        #cv2.waitKey(0)
        cv2.imshow(FileName, Hori)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    Img = cv2.resize(Img, (256, 256), cv2.INTER_AREA)

    if SegMask is not None:
        SegMask = cv2.resize(SegMask*255, (256, 256), cv2.INTER_AREA)
        SegMask = np.array((SegMask>30)*255, np.uint8)

    return Img, np.array( (X0, X1, Y0, Y1) ), SegMask

def PrepareSmalls():
    _image_files = glob.glob(f"{CrPath}finalres/*.png")

    Predict(rf'{CrPath}finaltest/', rf'{CrPath}finalres/', rf'{CrPath}models/', GrayScaled=True)
    for File in tqdm(_image_files):
        NewImg = ClearFields(cv2.imread(File, cv2.IMREAD_GRAYSCALE), FileName=File, Verbose=False)
        if not cv2.imwrite(File, NewImg):
            print(f'Ошибка записи {File}')

def GenerateMaskesAndZoom(Path='train', Zoom = True):
    dataset = EyeDataset(Path, ImgReadMode=cv2.IMREAD_GRAYSCALE)

    for sample in tqdm(dataset):
        # if Step < FileIndex*50: # аккуратненько чтоб не вникать в тонкости устройства dataset пропускаем уже сделанные позиции (использую при обрыве выполнения)
        #    Step+= 1
        #    continue

        m = sample["mask"][:, :, 1]
        File = dataset.CrFile

        basename = os.path.basename(File).split(".")[0]

        if Zoom:
            SegMask = cv2.resize(m, (256, 256),  interpolation = cv2.INTER_AREA)
            m = np.array((SegMask > 0.1), np.uint8)

            Img = cv2.resize(sample['image'], (256, 256), interpolation=cv2.INTER_AREA)

            if not cv2.imwrite(rf'{CrPath}{Path}/f{basename}.png', Img):
                print(f'Ошибка записи {File.split(".")[0] + "_m.png"}')

        if not cv2.imwrite(rf'{CrPath}{Path}/m{basename}.png', m * 255):
            print(f'Ошибка записи {File.split(".")[0] + "_m.png"}')



def PrepareTrain(Path='train', Verbose= False):
    dataset = EyeDataset(Path, ImgReadMode=cv2.IMREAD_GRAYSCALE)

    for sample in tqdm(dataset):
        # if Step < FileIndex*50: # аккуратненько чтоб не вникать в тонкости устройства dataset пропускаем уже сделанные позиции (использую при обрыве выполнения)
        #    Step+= 1
        #    continue

        m = sample["mask"][:, :, 1]
        Img = sample['image']

        File = dataset.CrFile
        NewImg, Disp, m = PrepareTest(Img, SegMask = m, Verbose=Verbose)
        if Verbose == False:
            basename = os.path.basename(File).split(".")[0]
            if not cv2.imwrite(rf'{CrPath}Res/f{basename}.png', NewImg):
                print(f'Ошибка записи {File}')
            if not cv2.imwrite(rf'{CrPath}Res/m{basename}.png', m):
                print(f'Ошибка записи {File.split(".")[0] + "_m.png"}')
            np.save(rf'{CrPath}Res/{basename}.npy', Disp)

def PrepareTests(Verbose, Path = f"{CrPath}finaltest_m/*.png"):
    _image_files = glob.glob(Path)

    for File in tqdm(_image_files):
        NewImg, Disp, _ = PrepareTest(cv2.imread(File, cv2.IMREAD_GRAYSCALE), FileName=File, Verbose=Verbose)
        if Verbose == False:
            if not cv2.imwrite(File, NewImg):
                print(f'Ошибка записи {File}')
            np.save(File.split('.')[0] + '.npy', Disp)

def DeleteDeletedMasks():
    maskes_files = glob.glob(f"{CrPath}Maskes/Bad/m*.png")

    for File in tqdm(maskes_files):
        basename = os.path.basename(File).split(".")[0][1:]

        try:
            os.replace(f"{CrPath}train/f{basename}.png", f"{CrPath}Maskes/Bad/f{basename}.png")
            print(f'Файл f{basename} перемещен')
        except:
            print(f'Файл f{basename} не найден')

        try:
            os.replace(f"{CrPath}train/m{basename}.png", f"{CrPath}Maskes/Bad/m{basename}.png")
            print(f'Файл m{basename} перемещен')
        except:
            print(f'Файл m{basename} не найден')

# Заносит картинки в numpy - файл
# Files м.б. массивом или списком номеров файлов. Тогда From - просто путь или путь и первая буква результата. Иначе From - маска (ex. C:/*.png)
def GenerateImgFile(From, To, Scale = (256, 256), Mode = cv2.IMREAD_GRAYSCALE, interpolation = cv2.INTER_AREA, MakeBinary = False, Files = None ):
    if Files is None:
        Files = glob.glob(From)
    else:
        FileList = [0]*len(Files)
        for i, f in enumerate(Files):
            fn = fr'{From}{Files}.png'
            FileList[i] = fn

        Files = FileList

    FileList = [None]* len(Files)

    i = 0
    for f in tqdm(Files):
        Img = cv2.imread(f, Mode)
        if Mode != cv2.IMREAD_GRAYSCALE:
            Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        Img = np.array(Img, dtype=np.ubyte)

        if Scale != None:
            if Img.shape[0] != Scale[1] or Img.shape[1] != Scale[0]:
                Img = cv2.resize(Img, Scale,  interpolation = interpolation)

                if interpolation == cv2.INTER_AREA and MakeBinary:
                    Img = np.array((Img > 30) * 255, np.uint8)

        Img = Img.reshape( (1, Img.shape[0], Img.shape[1], 1 if Mode == cv2.IMREAD_GRAYSCALE else 3))

        FileList[i] = Img

        i+= 1

    Res = np.concatenate(FileList)
    np.save(To, Res)

    return Res, Files
        # Img2 = cv2.resize(Img1, (32*12, 32*12))#,  interpolation = cv2.INTER_AREA)

def GenerateXYFile(From, To, Scale = (256, 256), Mode = cv2.IMREAD_GRAYSCALE, interpolation = cv2.INTER_AREA, MakeBinary = False ):
    Files = glob.glob(From)

    FileList = [None]* len(Files)
    MaskList = [None] * len(Files)

    i = 0
    for f in tqdm(Files):
        Img = cv2.imread(f, Mode)
        if Mode != cv2.IMREAD_GRAYSCALE:
            Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        Img = np.array(Img, dtype=np.ubyte)

        if Scale != None:
            Img = cv2.resize(Img, Scale,  interpolation = interpolation)

            if interpolation == cv2.INTER_AREA and MakeBinary:
                Img = np.array((Img > 30) * 255, np.uint8)

        Img = Img.reshape( (1, Img.shape[0], Img.shape[1], 1 if Mode == cv2.IMREAD_GRAYSCALE else 3))


        # Теперь маска
        MaskName = 'm' + os.path.basename(File).split(".")[0][1:]
        Img = cv2.imread(MaskName, Mode)

        Img = np.array(Img, dtype=np.ubyte)

        if Scale != None:
            Img = cv2.resize(Img, Scale, interpolation=interpolation)

            if interpolation == cv2.INTER_AREA and MakeBinary:
                Img = np.array((Img > 30) * 255, np.uint8)

        Img = Img.reshape((1, Img.shape[0], Img.shape[1], 1 if Mode == cv2.IMREAD_GRAYSCALE else 3))

        MaskList[i] = Img

        i+= 1

    ResX = np.concatenate(FileList)
    np.save(OutX, ResX)

    ResY = np.concatenate(FileList)
    np.save(OutX, ResX)

    return Res, Files
        # Img2 = cv2.resize(Img1, (32*12, 32*12))#,  interpolation = cv2.INTER_AREA)


def GenerateTrainXYFiles(From, To = None, Scale = None, Mode = cv2.IMREAD_GRAYSCALE, interpolation = cv2.INTER_AREA, MakeBinary = False ):
    if To == None:
        To = From
    GenerateImgFile(fr'{From}f*.png', fr'{To}testx2.npy', Scale = Scale)
    GenerateImgFile(fr'{From}m*.png', fr'{To}testy2.npy', Scale = Scale)

# Test Path0 и ResPath0 - это немодифицированные данные
# Test Path1 и ResPath1 - это модифицированные данные
def PredictLarge(TestPath0 = f'{CrPath}finaltest/', TestPath1 = f'{CrPath}finaltest_m/', ResPath0 = f'{CrPath}finalres/', ResPath1 = f'{CrPath}finalres_m/', ModelPath = f'{CrPath}models/', GrayScaled=True, Model=None, Mdf = True, Scale = None):
        # Сперва обработка модифицированных тестов
        if Mdf:
            TestPath = TestPath1
            ResPath = ResPath1
        else:
            TestPath = TestPath0
            ResPath = ResPath0

        try:
            Test = np.load(fr'{TestPath}test.npy')
            Files = []
            with open(fr'{TestPath}Files.lst', 'r') as filehandle:
                for line in filehandle:
                    # удалим заключительный символ перехода строки
                    currentPlace = line[:-1]

                    # добавим элемент в конец списка
                    Files.append(currentPlace)
        except: # создаем npy файл, содержащие выравненные до общего размера
            Test, Files = GenerateImgFile(fr'{TestPath}*.png', fr'{TestPath}test.npy', Scale = Scale)
            with open(fr'{TestPath}Files.lst', 'w') as filehandle:
                for listitem in Files:
                    filehandle.write('%s\n' % listitem)

        if Model == None:
            Modelfiles = glob.glob(f"{ModelPath}*.h5")
            ModelIndex = len(Modelfiles)

            if ModelIndex == 1:
                model = load_model(Modelfiles[0])
                #model.save_weights(r'D:/musor/W9.h5')


        else:
            model = Model
            ModelIndex = 1

        # Test = Test.reshape(Test.shape[0] * Test.shape[1], 128, 128, 1 if GrayScaled else 3)
        # Test = np.array(Test, np.float32)

        ModelCount = ModelIndex
        while ModelIndex > 0:

            if ModelCount > 1:
                model = load_model(Modelfiles[ModelIndex])

            ModelIndex -= 1

            StartPos = 0
            EndPos = 7

            while StartPos < len(Test):



                    ResTest = model.predict(Test[StartPos:EndPos], verbose=False)[:, :, :, 0:1]

                    for Img, F in zip(ResTest, Files[StartPos:EndPos]):
                        basename = os.path.basename(F)
                        FileName = os.path.splitext(basename)[0]

                        if Mdf:
                            Shapes = np.load(fr'{TestPath1}npy/{FileName}.npy')
                            Res = cv2.resize(Img,(Shapes[3] - Shapes[2] + 1, Shapes[1] - Shapes[0] + 1), interpolation = cv2.INTER_LANCZOS4 )

                            Maska = np.zeros( (1232, 1624), dtype = np.float32)
                            Maska[Shapes[0]:Shapes[1]+1, Shapes[2]:Shapes[3] +1] = Res
                        else:
                            Maska = cv2.resize(Img,(1624, 1232), interpolation = cv2.INTER_NEAREST)

                        Res = np.array(Maska>0.30, dtype = np.uint8) * 255
                        cv2.imwrite(fr'{ResPath}{FileName}.png', Res)


                    StartPos = EndPos
                    print(StartPos, 'из', len(Test), end='\r')

                    EndPos += 7
        print('')


def PredictLargeAnsb(TestPath0=f'{CrPath}finaltest/', TestPath1=f'{CrPath}finaltest_m/', ResPath0=f'{CrPath}finalres/',
                 ResPath1=f'{CrPath}finalres_m/', ModelPath=f'{CrPath}models/', GrayScaled=True, Model=None, Mdf=True):
    # Сперва обработка модифицированных тестов
    if Mdf:
        TestPath = TestPath1
        ResPath = ResPath1
    else:
        TestPath = TestPath0
        ResPath = ResPath0

    try:
        Test = np.load(fr'{TestPath}test.npy')
        Files = []
        with open(fr'{TestPath1}Files.lst', 'r') as filehandle:
            for line in filehandle:
                # удалим заключительный символ перехода строки
                currentPlace = line[:-1]

                # добавим элемент в конец списка
                Files.append(currentPlace)
    except:
        Test, Files = GenerateImgFile(fr'{TestPath}*.png', fr'{TestPath}test.npy')
        with open(fr'{TestPath}Files.lst', 'w') as filehandle:
            for listitem in Files:
                filehandle.write('%s\n' % listitem)

    if Model == None:
        Modelfiles = glob.glob(f"{ModelPath}*.h5")
        ModelIndex = len(Modelfiles)

        if ModelIndex == 1:
            model = load_model(Modelfiles[0])



    else:
        model = Model
        ModelIndex = 1

    # Test = Test.reshape(Test.shape[0] * Test.shape[1], 128, 128, 1 if GrayScaled else 3)
    # Test = np.array(Test, np.float32)

    ModelCount = ModelIndex
    while ModelIndex > 0:

        if ModelCount > 1:
            model = load_model(Modelfiles[ModelIndex])

        ModelIndex -= 1

        StartPos = 0
        EndPos = 7

        while StartPos < len(Test):

            ResTest = model.predict(Test[StartPos:EndPos], verbose=False)[:, :, :, 0:1]

            for Img, F in zip(ResTest, Files[StartPos:EndPos]):
                basename = os.path.basename(F)
                FileName = os.path.splitext(basename)[0]

                if Mdf:
                    Shapes = np.load(fr'{TestPath1}/npy/{FileName}.npy')
                    Res = cv2.resize(Img, (Shapes[3] - Shapes[2] + 1, Shapes[1] - Shapes[0] + 1),
                                     interpolation=cv2.INTER_LANCZOS4)

                    Maska = np.zeros((1232, 1624), dtype=np.float32)
                    Maska[Shapes[0]:Shapes[1] + 1, Shapes[2]:Shapes[3] + 1] = Res
                else:
                    Maska = cv2.resize(Img, (1624, 1232), interpolation=cv2.INTER_NEAREST)

                Res = np.array(Maska > 0.3, dtype=np.uint8)
                cv2.imwrite(fr'a{ResPath}{FileName}.png', Res)

            StartPos = EndPos
            print(StartPos, 'из', 301, end='\r')

            EndPos += 7
    print('')

def dice_coef_np(y_true, y_pred):
    return (2. * np.sum(y_true * y_pred) + 1.) / (np.sum(y_true) + np.sum(y_pred) + 1.)

def TestMaskes(Calculated=f'{CrPath}train_m/*.png', Tested = f'{CrPath}train/', Verbose = False):
    maskes_files = glob.glob(Calculated)

    Res0 = np.empty((len(maskes_files),2))
    Res1 = np.empty((len(maskes_files), 2))

    Ind = 0

    for File in tqdm(maskes_files):
        basename = os.path.basename(File)
        Index = basename.split(".")[0]


        Img0 = cv2.imread(File, cv2.IMREAD_GRAYSCALE) > 128
        Img1 = cv2.imread(fr'{Tested}m{basename}', cv2.IMREAD_GRAYSCALE)

        try:
            Img1 = Img1 > 128
        except:
            print(basename)
        try:
            V0 = dice_coef_np(Img0, Img1)
            V1 = (Img0.sum() - (Img0*Img1).sum())/(0.00001 + (Img0*Img1).sum())
        except:
            print(basename)

        Res0[Ind,:] = (int(Index), V0)
        Res1[Ind,:] = (int(Index), V1)

        Ind+= 1

        if Verbose:
            Img2 = Img0 // 2
            Hori = np.concatenate((255 * Img1, 255 * (Img1 * (1 - Img0) + Img2)), axis=1)
            Hori = cv2.resize(Hori, (1024, 1024))
            # concatenate image Vertically
            cv2.imshow(basename, Hori)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    Sort = np.argsort(Res0[:, 1],0)
    Res0 = Res0[Sort]
    #0.35
    Sort1 = np.argsort(Res1[:, 1],0)
    Res1 = Res1[Sort1]

    return Res0, Res1

if __name__ == '__main__':
    #Y1 = np.load(f'{CrPath}finaltest_m/test.npy')#PrepareSmalls()
    #cv2.imwrite(r'E:/musor/Test.png', Y1[0])
    #Img1 = cv2.imread(r'E:/musor/m99.png', cv2.IMREAD_GRAYSCALE)
    #Img2 = cv2.resize(Img1, (32*12, 32*12))#,  interpolation = cv2.INTER_AREA)
    #Img2 = (Img2>30)*255
    #cv2.imwrite(r'E:/musor/m99s.png', Img2)
    #Img2 = cv2.imread(r'E:/musor/debugBlack.png', cv2.IMREAD_GRAYSCALE)
    #PrepareTest(cv2.imread(r'E:/w/Hacatons/Vladik/finaltest/862.png', cv2.IMREAD_GRAYSCALE), Verbose=True)

    #GenerateMaskesAndZoom('Train2', Zoom= False)
    #GenerateTrainXYFiles(fr'{CrPath}train/')
    #PrepareTrain("LTrain",Verbose = False)
    #DeleteDeletedMasks()
    '''
    X = np.load(fr'{CrPath}XTrainTr.npy')
    Sz = len(X)//2
    np.save(fr'{CrPath}XTrainTr0.npy', X[:Sz])
    np.save(fr'{CrPath}XTrainTr1.npy', X[Sz:])
    X = np.load(fr'{CrPath}YTrainTr.npy')
    np.save(fr'{CrPath}YTrainTr0.npy', X[:Sz])
    np.save(fr'{CrPath}YTrainTr1.npy', X[Sz:])
    '''
    '''
    # 1. Вручную распаковываем тестовый набор в папку finaltest_m
    
    # 2. Создаем обрезанный тестовый набор в finaltest_m
    PrepareTests(Path=f"{CrPath}finaltest_m/*.png", Verbose=False)  # обрезаем файлы
    # Этот набор можно использовать для обучения сети. А, например, добавив слева к имени 99990, объединить с основным набором,
    # и проводить обучение как на обрезанном, так и на основном наборе 
    # 3. Из finaltest_m в finaltest_m/npy вручную переносим файлы npy
    
    # 4. Теперь можно протестировать результат модели, помещенной в папку models (не более одной модели!)
    PredictLarge(Mdf=True)
    # Набор масок для отправки будет собран в папке finalres_m
    
    # Далее. Не обязательно, но удобно
    # 4. Создаем файлы масок в train
    GenerateMaskesAndZoom(Path='train', Zoom=False)
    '''


    # По умолчанию файл настроен на проверку файла модели, который находится в каталоге models
    PredictLarge(Mdf=True)
