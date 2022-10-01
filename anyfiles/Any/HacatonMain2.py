from UNetGenetic import TUNetGenetic
import numpy as np
import random
from NetFromHromosom import TNNHromosom, BinClassification, dice_coef
from tensorflow.keras.models import Model, load_model # Импортируем модели keras: Model
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation, SpatialDropout2D, MaxPooling2D, AveragePooling2D, Conv2D, BatchNormalization # Импортируем стандартные слои keras
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

GrayScaled = True

#random.seed(1)

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

    image = np.array(image, dtype=np.ubyte)#.reshape

    return image



def GenerateTrain(path, GrayScaled = True):
    _image_files = glob.glob(f"{path}/*.png")

    Ress = []
    Res = np.array([], dtype=np.ubyte)
    Index = 0
    HallIndex = 0
    FileIndex = 2
    Step = 0

    for f in tqdm(_image_files):
        Img = read_image(f, GrayScaled)

        #if Step < FileIndex*50: # аккуратненько чтоб не вникать в тонкости устройства dataset пропускаем уже сделанные позиции (использую при обрыве выполнения)
        Step+= 1
            #continue

        Shape = Img.shape

        if Shape[0] != 1232 or Shape[1] != 1624:
            Img = cv2.resize(Img, (1624, 1232) )


            #cv2.imshow('', Img * (1 - m))
            # cv2.imshow('', Img)
            #cv2.waitKey(0)

        #continue

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

                Res = np.append(Res, Img[Left:Left + 128, Top : Top + 128])

        Index += 1

        if Index == 18 or HallIndex == len(_image_files)-1:
            L = len(Res) // (128 * 128)
            if L *  (128 * 128) != len(Res):
                print('Error!')
            Res = np.array(Res).reshape( (L , 128, 128, 1))

            #np.save(f'numpy/train{FileIndex}', Res)

            Ress.append(Res)

            FileIndex+= 1
            #print(f'Сформирован файл /numpy/train{Index}.np')

            Index = 0

            Res = np.array([], dtype = np.ubyte)

        HallIndex+= 1

    return np.concatenate(Ress), _image_files


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

        SumRes+= Res

    Res = (1 - (np.array(SumRes / len(Slides)) >= 0.5)).reshape((1232, 1624, 1))

    return Res

# X в формате [N, 128, 128, 1]
def PredictList(X, GrayScaled = True, Model = None):
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

def Predict(path, ResPath, ModelPath, GrayScaled = True, Model = None):
    try:
        Test = np.load(fr'{path}test.npy')
        #np.save(fr'{path}test.npy', Test)
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


    custom_objects = {"dice_coef": dice_coef}

    if Model == None:
        files = glob.glob(f"{ModelPath}*.h5")
        for M in files:

            model = load_model(M, custom_objects=custom_objects)
    else:
        model = Model
    #Test = Test.reshape(Test.shape[0] * Test.shape[1], 128, 128, 1 if GrayScaled else 3)
    #Test = np.array(Test, np.float32)

    ResTest = model.predict(Test, verbose=True)[:, :, :, 0:1]

    Shape = list(ResTest.shape)

    ResTest = ResTest.reshape((Shape[0] // (12*16), 12, 16, Shape[-3], Shape[-2], Shape[-1]))

    Index = 0

    for Img, F in zip(ResTest, Files):
        Res = RestoreSmallImg(Img.reshape((1, 12, 16, 128, 128)))

        Res = np.array(Res, dtype = np.byte)
        basename = os.path.basename(F)
        FileName = os.path.splitext(basename)[0]
        #cv2.imwrite(fr'{ResPath}dRAW_{os.path.splitext(basename)[0]}.png', Res*255)
        #cv2.imwrite(fr'{ResPath}{os.path.splitext(basename)[0]}.png', Res)
        cv2.imwrite(fr'{ResPath}dRAW_{FileName}.png', Res * 255)
        cv2.imwrite(fr'{ResPath}{FileName}.png', Res)

        Index+= 1

def GenerateTestImg(CrPath, ResPath, Files):
    Data = np.load(rf'{CrPath}numpy/train.npy')

    InputY = Data[:, 1]

    InputY.reshape( (len(InputY)//(12*16), 12, 16, 128, 128) )

    for Img, F in zip(ResTest, Files):
        Res = RestoreSmallImg(Img.reshape((1, 12, 16, 128, 128)))

        Res = np.array(Res, dtype=np.byte)
        basename = os.path.basename(F)
        cv2.imwrite(fr'mDRAW_{ResPath}{os.path.splitext(basename)[0]}.png', Res)
        cv2.imwrite(fr'm_{ResPath}{os.path.splitext(basename)[0]}.png', Res)

if __name__ == '__main__':
    #model = unetWithMask()
    #W = np.load(rf'{CrPath}cr.npy', allow_pickle=True)
    #model.set_weights(W)
    Predict(rf'{CrPath}finaltest/', rf'{CrPath}finalres/', rf'{CrPath}models/', GrayScaled = True)

    Data = np.load(rf'{CrPath}numpy/train.npy')
    Shape = Data.shape
    #Data = Data.reshape( (Shape[0]*Shape[1]*Shape[2], Shape[3], Shape[4], Shape[5]))
    np.random.shuffle(Data)
    NTest = len(Data)//10
    Xlen = len(Data) - NTest

    InputX = Data[:-NTest, 0]
    XShape = list(InputX.shape)
    XShape.append(1)
    InputX = InputX.reshape(XShape)

    InputY = Data[:-NTest, 1]
    YShape = list(InputY.shape)
    YShape.append(1 if GrayScaled else 3)
    InputY = InputY.reshape(YShape)

    TestX = Data[-NTest:, 0]
    TestXShape = list(TestX.shape)
    TestXShape.append(1 )
    TestX = TestX.reshape(TestXShape)

    TestY = Data[-NTest:, 1]
    TestYShape = list(TestY.shape)
    TestYShape.append(1 if GrayScaled else 3)
    TestY = TestY.reshape(TestYShape)

    YSum = np.sum(InputY.reshape((InputY.shape[0], InputY.shape[1] * InputY.shape[2])), 1)
    Ind = YSum.argsort()[-1024:]
    InputX = InputX[Ind]
    InputY = InputY[Ind]
    InputY1 = 1 - InputY

    InputY = np.concatenate([InputY, InputY1], 3)

def PackImg(Path):
        dataset = EyeDataset(Path, ImgReadMode=cv2.IMREAD_GRAYSCALE)
        Ress = []
        Res = np.array([], dtype=np.ubyte)
        Index = 0
        FileIndex = 2
        Step = 0
        for sample in tqdm(dataset):

            if Step < FileIndex * 50:  # аккуратненько чтоб не вникать в тонкости устройства dataset пропускаем уже сделанные позиции (использую при обрыве выполнения)
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

                    Res = np.append(Res, Img[Left:Left + 128, Top: Top + 128])

            Index += 1


        L = len(Res) // (128 * 128)
        Res = np.array(Res).reshape((L, 128, 128, 1))

        return Res



class THromosomCallback(Callback):
    def __init__(self):
        self.State = [0, 0]  # лучшее значение, его позиция
        self.BestW = None

    def on_epoch_end(self, epoch, logs=None):
        if (logs['dice_coef'] == None) or (logs['dice_coef'] < 0.01) :
            print('Abort learning, dice_coef=', logs['dice_coef'])
            #print('Abort learning, acc=', logs['accuracy'], file=f)

            self.model.stop_training = True
            return

        CR = logs['val_dice_coef']
        if CR > self.State[0]:
            self.State[0] = CR
            self.State[1] = epoch

            self.BestW = self.model.get_weights()
            print('Сохранено наибольшее значение')

            if epoch % 5 == 5:
                np.save(rf'{CrPath}/cr.npy', self.BestW)
        elif epoch % 5 == 5:
            np.save(rf'{CrPath}/cr.npy', self.model.get_weights())

if __name__ == '__main__':
    CrCallback = THromosomCallback()
    model = unetWithMask()
    #W = np.load(rf'{CrPath}cr.npy', allow_pickle=True)
    #model.set_weights(W)


    history = model.fit(InputX, InputY, epochs=20, batch_size=16, validation_data = (TestX, TestY), callbacks = [CrCallback])

    Test = PackImg(Path)
    model.set_weights(CrCallback.BestW)
    np.save(rf'{CrPath}/best.npy')
    model.save_model(rf'{CrPath}/BestModel.h5')
    Res = model.evaluate(TestX, TestY, verbose=True)
    ResTest = model.predict(Test, verbose=True)


    Res = np.array(Res).reshape((L, 128, 128, 1))
    for i in range(L // (12*16)):
        Img = RestoreSmallImg(Res[i*(12*16):(i+1)*i*(12*16)])

    Genetic = TUNetGenetic(InputX, InputY, TestX, TestY, 1, MaxLevels = 3)
    Genetic.Start()

