import numpy as np
import random
from tensorflow.keras.models import Model, load_model # Импортируем модели keras: Model
from tensorflow.keras.layers import Input, Rescaling, Conv2DTranspose, concatenate, Add, Activation, SpatialDropout2D, MaxPooling2D, AveragePooling2D, Conv2D, BatchNormalization # Импортируем стандартные слои keras
from tensorflow.keras import backend as K # Импортируем модуль backend keras'а
from tensorflow.keras.optimizers import Nadam, Adam # Импортируем оптимизатор Adam
from tensorflow.keras import utils # Импортируем модуль utils библиотеки tensorflow.keras для получения OHE-представления
from keras import regularizers
from keras.callbacks import Callback
import tensorflow as tf
import os
from tqdm.auto import tqdm
import glob
import cv2
import gc
from tqdm.auto import tqdm
import json
from torch.utils.data import Dataset
from ResultProcessing import TestMaskes

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

def GenerateTrainFiles(From, To, Postfix, BadList = None):
    Files = glob.glob(rf"{CrPath}{From}/f*.png")

    if BadList is not None:
        Bads = len(BadList)
    else:
        Bads = 0
        BadList = []

    Len1 = (2*len(Files) - Bads) // 2
    Len0 = 0

    Ind = np.arange(len(Files))
    np.random.shuffle(Ind)

    for FileID in range(2):
        ResX = [None] * (Len1 - Len0)
        ResY = [None] * (Len1 - Len0)
        i = 0
        for FileInd in tqdm(Ind[Len0:Len1]):
            File = Files[FileInd]
            File = File.replace('\\f', '/f')
            mPath = File.replace('/f', '/m')

            if not (File in BadList or mPath in BadList):

                image = cv2.imread(File, cv2.IMREAD_GRAYSCALE)
                image = image.reshape( (1, 256, 256, 1) )

                m = cv2.imread(mPath, cv2.IMREAD_GRAYSCALE)


                #cv2.imwrite(mPath, m)
                if m is not None:
                    m = np.array(255 * (m > 120), dtype=np.uint8)
                    m = m.reshape((1, 256, 256, 1))
                else:
                    m = np.zeros((1, 256, 256, 1), dtype = np.uint8)

                ResX[i] = image
                ResY[i] = m
                i+=1
            #else:
                #print(6)

        X = np.concatenate(ResX[:i])
        np.save(rf'{To}/X{Postfix}{FileID}.npy', X)
        X = None


        Y = np.concatenate(ResY[:i])
        Y //= 255
        Y1 = 1 - Y

        Y = np.concatenate([Y, Y1], 3)
        np.save(rf'{To}/Y{Postfix}{FileID}.npy', Y)
        Y = None
        gc.collect()

        Len0 = i
        Len1 = len(Files)

def GenerateTestFiles(From, To, Postfix):
    Files = glob.glob(rf"{CrPath}{From}/*.png")

    ResX = [None]*len(Files)
    i = 0

    for File in tqdm(Files):
        image = cv2.imread(File, cv2.IMREAD_GRAYSCALE)
        image = image.reshape( (1, 256, 256, 1) )

        ResX[i] = image
        i+=1

    X = np.concatenate(ResX)

    np.save(To + '/X' + Postfix + '.npy', X)

    with open(To + '/FileList.lst', 'w') as filehandle:
        for listitem in Files:
            filehandle.write('%s\n' % listitem)

#NB! From =- имя файла. To - имя файла без расширения (концовка файла и расширение потом добавятся автоматом
def TruncImg(From, To, IsMask):
    basename = os.path.basename(From)
    #Path = From[:-len(basename)]
    FileName = os.path.splitext(basename)[0]

    image = cv2.imread(From, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print('Не обнаружено', FileName)
        return

    image = cv2.resize(image, (256*6, 256*4), interpolation = cv2.INTER_AREA)

    image = image.reshape( (4, 256, 6, 256))

    for x in range(4):
        for y in range(6):
            # Анализируем, что получилось. Идея такова - для того, чтобы файл пошел в обучение, надо, чтобы или была маска,
            # или на файле было что-то интересное (высокая вариативность)
            # Спорные файлы помечаются '_' на конце.
            # Пустые маски не храним, они создаются на лету при необходимости
            Flag = ''

            CrImage = image[x, :, y, :]
            cv2.imwrite(fr'{To}_{x}_{y}{Flag}.png', CrImage)



def GenerateDataset(From, FileIDList, N, To = 'FullTrain/Res'):
    Src = fr'{CrPath}{From}/'
    Dst = fr'{CrPath}{To}/'

    #_image_files = glob.glob(rf"{Src}m*.png")

    for ID in tqdm(FileIDList[-N:]):
        FileName = str(ID) + '.png'
        FileName2 = fr'99990{ID}.png'
        TruncImg(fr'{Src}m{FileName}', fr'{Dst}m{FileName}', IsMask=True)
        TruncImg(fr'{Src}{FileName}', fr'{Dst}f{FileName}', IsMask=False)
        TruncImg(fr'{Src}m{FileName2}', fr'{Dst}m{FileName2}', IsMask=True)
        TruncImg(fr'{Src}{FileName2}', fr'{Dst}f{FileName2}', IsMask=False)



# From и To - простые названия каталогов, без путей.
def GenerateTestDataset(From = 'finaltest_m', To = 'finaltest_tr'):
    Src = fr'{CrPath}{From}/'
    Dst = fr'{CrPath}{To}/'

    _image_files = glob.glob(rf"{Src}*.png")

    Indexes = np.empty( (len(_image_files)), dtype= np.int16 ) #Индексация позиций массива
    Index = 0
    for File in tqdm(_image_files):
        basename = os.path.basename(File)
        FileName = os.path.splitext(basename)[0]

        TruncImg(File, fr'{Dst}{FileName}', IsMask = False)
        Indexes[Index] = int(FileName)

        Index += 1

    np.save(rf'{Dst}Indexes.npy', Indexes)

def TestTrainFile(Path = CrPath, ID = 0):
    XTrain = np.load(fr'{Path}XTrainTr{ID}.npy')
    YTrain = np.load(fr'{Path}YTrainTr{ID}.npy')

    while True:
        ID = random.randint(0, len(XTrain))

        X = XTrain[ID]
        Y = YTrain[ID]*255

        X1 = X | Y[:,:,0:1]

        Hori = np.concatenate((X, X1, Y[:,:,0:1]), axis=1)

        # concatenate image Vertically
        cv2.imshow(str(ID), Hori)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#def GenerateTestFile():

def TestPare(s, m, List):
    Img0 = cv2.imread(s, cv2.IMREAD_GRAYSCALE)
    if Img0 is None:
        return None

    Img1 = cv2.imread(m, cv2.IMREAD_GRAYSCALE)

    if Img1.sum() < 2550 and np.std(Img0) < 10:
        List.append(s)
        List.append(m)

    return List

def TestTruncDataset(TruncPath):
    List = []
    for i in tqdm(range(783)):
        R = 0
        for x in range(4):
            for y in range(6):
                R = TestPare(fr'{TruncPath}f{i}_{x}_{y}.png', fr'{TruncPath}m{i}_{x}_{y}.png', List)

                if R is None:
                    break


                List = TestPare(fr'{TruncPath}f99990{i}_{x}_{y}.png', fr'{TruncPath}m99990{i}_{x}_{y}.png', R)

            if R is None:
                #print('Пропускаю', i)
                break


    return List

# Если Temp = None, то данные уже парезаны в From, а также создан файл npy, собравший все изображения
# From и To - простые названия каталогов, без путей.
# Truncated - каталог с обрезанными картинками. В нем д.б. папка npy с координатами обрезки - они нужны для восстановления
# Если не указана, считается, что данные немодифицированы
def Predict(From = 'finaltest_tr', Truncated = 'finaltest_m', Temp = None, Models = 'models', FirstCall = False):
    Mdf = Truncated is not None
    if FirstCall:
        print('Нарезка файлов')
        GenerateTestDataset(From, Temp)
        print('Создаем npy файл с тестами')
        GenerateTestFiles(Temp, Temp, 'Test')
    else:
        Temp = From

    Src = fr'{CrPath}{From}/'
    Dst = fr'{CrPath}{Temp}/'

    FileList = []

    print('Загружаем тестовые данные')
    X = np.load(fr'{Dst}XTest.npy')

    Files = {} # хранится номер картинки, X, Y разбивки
    # откроем файл и считаем его содержимое в список
    with open(Temp + '/FileList.lst', 'r') as filehandle:
        for ID, line in enumerate(filehandle):
            # удалим заключительный символ перехода строки
            f = line[:-1]

            basename = os.path.basename(f)
            FileName = os.path.splitext(basename)[0]
            FileID = FileName[:-4]
            FileX = int(FileName[-3])
            FileY = int(FileName[-1])

            FileInfo = Files.get(FileID)
            if FileInfo is None:
                FileInfo = np.empty( (4, 6), dtype = np.uint8)

                Files[FileID] = FileInfo

            FileInfo[FileX, FileY] = ID


    print('Загружаем модель')
    Modelfiles = glob.glob(fr'{CrPath}{Models}/*.h5')
    ModelIndex = len(Modelfiles)

    for M in Modelfiles:
        model = load_model(M)

    print('Сегментация')

    Res = np.empty_like(X)

    N = len(X)//3
    
    for i in tqdm(range(N)):
        Left = i * 3
        Right = Left + 3

        Res[Left:Right] = model.predict(X[Left:Right], verbose = False)[:, :, :, 0:1]

        Left = Right
        Right = Left + 3

    #Img = np.empty( (1024, 1024) )

    for FileID, Data in Files.items():
        L0 = np.column_stack( Res[Data[0, 0:5]])
        L1 = np.column_stack( Res[Data[1, 0:5]])
        L2 = np.column_stack( Res[Data[2, 0:5]])
        L3 = np.column_stack( Res[Data[3, 0:5]])
        Img = np.row_stack( (L0,L1,L2,L3) )

        if Mdf:

            Shapes = np.load(fr'{Truncated}/npy/{FileID}.npy')
            Res = cv2.resize(Img, (Shapes[3] - Shapes[2] + 1, Shapes[1] - Shapes[0] + 1),
                             interpolation=cv2.INTER_AREA)

            Maska = np.zeros((1232, 1624), dtype=np.float32)
            Maska[Shapes[0]:Shapes[1] + 1, Shapes[2]:Shapes[3] + 1] = Res
        else:
            Maska = cv2.resize(Img, (1624, 1232), interpolation=cv2.INTER_AREA)

        Res = np.array(Maska > 0.30, dtype=np.uint8) * 255
        cv2.imwrite(fr'{Temp}{FileID}.png', Res)



'''
# Этап первый. Готовим файлы обучающего набора
List0, _ = TestMaskes(Calculated=f'{CrPath}train_m/*.png', Tested = f'{CrPath}train/')
GenerateDataset('FullTrain', np.array(List0[:,0], dtype = np.int32), (List0[:,1]>0.45).sum())
GenerateTrainFiles('FullTrain/Res', 'FullTrain/Res', 'FullTrain', BadList = TestTruncDataset(fr'{CrPath}FullTrain/Res/'))
TestTruncDataset('FullTrain/Res/')

#разбивает уже обрезанные в finaltest_m файлы на 24 части, и заносит в фай  л тестирования.
Predict(FirstCall = True, From = 'finaltest_m', Temp = 'finaltest_tr')
'''


#TestTrainFile(ID= 0)



#TruncImg(fr'{CrPath}train2/7.png', fr'{CrPath}train/Res/')
#GenerateTestDataset()

Predict(From = 'finaltest_tr')

