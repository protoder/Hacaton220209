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
            Img = cv2.resize(Img, (1624, 1232))

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

if __name__ == '__main__':
    _image_files = glob.glob(f"{CrPath}finalres/*.png")

    Predict(rf'{CrPath}finaltest/', rf'{CrPath}finalres/', rf'{CrPath}models/', GrayScaled = True)
    for File in tqdm(_image_files):
        NewImg = ClearFields(cv2.imread(File, cv2.IMREAD_GRAYSCALE), FileName = File, Verbose = False)
        if not cv2.imwrite(File, NewImg):
            print(f'Ошибка записи {File}')

