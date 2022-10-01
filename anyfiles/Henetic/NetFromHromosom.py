from tensorflow.keras.models import Model # Импортируем модели keras: Model
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Activation, SpatialDropout2D, MaxPooling2D, AveragePooling2D, Conv2D, BatchNormalization # Импортируем стандартные слои keras
from tensorflow.keras import backend as K # Импортируем модуль backend keras'а
from tensorflow.keras.optimizers import Adam # Импортируем оптимизатор Adam
from tensorflow.keras import utils # Импортируем модуль utils библиотеки tensorflow.keras для получения OHE-представления

import matplotlib.pyplot as plt # Импортируем модуль pyplot библиотеки matplotlib для построения графиков
from tensorflow.keras.preprocessing import image # Импортируем модуль image для работы с изображениями
import numpy as np # Импортируем библиотеку numpy
from sklearn.model_selection import train_test_split
import time
import random
import os # Импортируем библиотеку os для раоты с фаловой системой
from PIL import Image # импортируем модель Image для работы с изображениями
from keras import regularizers
from  Settings import *

def dice_coef(y_true, y_pred):
    R = (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

    if R == None:
        print(K.sum(y_true * y_pred), K.sum(y_true) + K.sum(y_pred))
        #raise('K = None')
    return R # Возвращаем площадь пересечения деленную на площадь объединения двух областей


'''
    Изначально все инициализируется значениями случайного генератора от 0 до 252
    Формат зависит от типа сети.
    Всегда заголовок сети начинается с:

        0 Кол-во слоев
        1 Оптимизатор
        2 - (Batch Size + 2) * 4? (0..251)
        3 Тип входа - может быть пропущен. Интерпретируется произвольно
        3 или 4 SoftMax/Sigmoid если задан признак Классификация или БинарнаяКлассификация
        3 или 4 или 5 Тип сети, если Тип сети Auto
        3 или 4 или 5 или 6 выходной слой
        
        Наш вариант:
        0 Кол-во слоев
        1 Оптимизатор
        2 (Batch Size + 2) * 4? (0..251) - не используется
        3 SoftMax/Sigmoid - не используется (вместо него - активация выходного слоя)
        4 Размер входных данных (473 + 86*х)
        5-8 - запас на будущее
        9-13 - выходной слой
        
    Выходной сверточный слой:
        0 Activation (расширенный вариант, с softmax)    
        1 kernel_regularizer (0..2)  (0..83 - None, 84 - 167 - L1, 168 - 252 - L2. Число указывает коэффициент. 0.005 + (N+1)/500. ТО есть максимум 0.088)
        2 Activity-regularizer
        3, 4 WinX, WinY - (0..7)+1
        
    Слой - задает Conv слой, Dence или Conv2DTranspose
        0 Neyrons - размер, (0 - 251) * 4. Максимум зависит от слоя
        1 Activation
        2 ActivationAfterNorm
        3 BatchNorm и DropOut (или SpetialDropout2D) 0..251. 84..168 BatchNorm. >168 (DropOut - 1000)* 0.0001
        4 kernel_regularizer (0..2)  (0..83 - None, 84 - 167 - L1, 168 - 252 - L2. Число указывает коэффициент. 0.005 + (N+1)/500. ТО есть максимум 0.088)
        5 Activity-regularizer
        6, 7 Для Conv еще WinX, WinY - (0..7)+1

    Блок -
        Заголовок блока
            0 Размер (0..2) = 1
            1 Вид Pooling - Max, Avg или Strides на последнем слое блока
            2 Проброс не прямой, а через слой
            
        Conv слой1
        Conv слой2
        Conv слой3
        TransposeБлок
        1 Conv слой для пробросов

'''
LastLayerSize = 5
LastLayerPos = 9
NetCaptionSize = LastLayerPos + LastLayerSize
FirstBlockPos = NetCaptionSize
BlockCaptionSize = 3
BlockCount = 5
LayerSize = 8
BlockSize = LayerSize*4 + BlockCaptionSize
RefLayersDisp = LayerSize*3 + BlockCaptionSize

EncoderStart = FirstBlockPos + BlockSize * BlockCount
TransposeLayerSize = 6
EncoderBlockCaptionSize = 1
EncoderBlockSize = LayerSize*3 + EncoderBlockCaptionSize + TransposeLayerSize
EncoderLayersStart = EncoderBlockCaptionSize + TransposeLayerSize






Auto = 0
Other = 255
UNet = 1

Classificator = 1
BinClassification = 2

DenseLayer = 1
Conv2DLayer = 2

Relu = 0
Elu = 1
Selu = 2
Tanh = 3
Sigmoid = 4
activation_list = ['relu', 'elu', 'selu', 'tanh', 'sigmoid']
out_activation_list = ['sigmoid', 'softmax']

Sgd = 0
Rmsprop = 1
Adagrad = 2
Adadelta = 3
Adam = 4
Adamax = 5
Nadam = 6
optimizer_list = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']

BatchNorm = 1
DropOut = 2


MaxPooling = 0
AvgPooling = 1
Strides = 2

class TNNHromosom():
    def __init__(self, MaxLevels, NetType = UNet, ResType = Classificator):
        self.NetType = NetType
        self.ResType = ResType

        self.MaxLevels = MaxLevels;

        self.MinNeyrons = 4 # минимальное кол-во нейронов деленое на self.KNeyrons
        self.KNeyrons = 8 if Debug == 0 else 1 # при вызове значение делится на 2

        self.HasInputType = False # имеется ли поле типа входа в описании сети

        if NetType == UNet:
            self.Mask = (   9999999999999999, 9999999999999999, # 1-st поля хромосомы служебные: Признак измененности, ID
                            MaxLevels, len(optimizer_list), 252, 2, 252, 252, 252, 252, 252,# заголовок сети
                            len(out_activation_list), 252, 252, 8, 8,# описание выходного блока
                            3, 3, 2, # загоовок блока 0
                            16, len(activation_list), 2, 252, 252, 252, 8, 8,
                            16, len(activation_list), 2, 252, 252, 252, 8, 8,
                            16, len(activation_list), 2, 252, 252, 252, 8, 8,
                            16, len(activation_list), 2, 252, 252, 252, 8, 8,
                            3, 3, 2,  # загоовок блока 1
                            32, len(activation_list), 2, 252, 252, 252, 8, 8,
                            32, len(activation_list), 2, 252, 252, 252, 8, 8,
                            32, len(activation_list), 2, 252, 252, 252, 8, 8,
                            32, len(activation_list), 2, 252, 252, 252, 8, 8,
                            3, 3, 2,  # загоовок блока 2
                            64, len(activation_list), 2, 252, 252, 252, 8, 8,
                            64, len(activation_list), 2, 252, 252, 252, 8, 8,
                            64, len(activation_list), 2, 252, 252, 252, 8, 8,
                            64, len(activation_list), 2, 252, 252, 252, 8, 8,
                            3, 3, 2,  # загоовок блока 3
                            128, len(activation_list), 2, 252, 252, 252, 8, 8,
                            128, len(activation_list), 2, 252, 252, 252, 8, 8,
                            128, len(activation_list), 2, 252, 252, 252, 8, 8,
                            128, len(activation_list), 2, 252, 252, 252, 8, 8,
                            3, 3, 2,  # загоовок блока 3
                            252, len(activation_list), 2, 252, 252, 252, 8, 8,
                            252, len(activation_list), 2, 252, 252, 252, 8, 8,
                            252, len(activation_list), 2, 252, 252, 252, 8, 8,
                            252, len(activation_list), 2, 252, 252, 252, 8, 8,

                            # Дальше пошли блоки энкодера
                            3,  # загоовок блока декодера 3
                            252, len(activation_list), 2, 252, 252, 252,
                            252, len(activation_list), 2, 252, 252, 252, 8, 8,
                            252, len(activation_list), 2, 252, 252, 252, 8, 8,
                            252, len(activation_list), 2, 252, 252, 252, 8, 8,
                            3,   # загоовок блока декодера 3
                            128, len(activation_list), 2, 252, 252, 252,
                            128, len(activation_list), 2, 252, 252, 252, 8, 8,
                            128, len(activation_list), 2, 252, 252, 252, 8, 8,
                            128, len(activation_list), 2, 252, 252, 252, 8, 8,
                            3,   # загоовок блока декодера 2
                            64, len(activation_list), 2, 252, 252, 252,
                            64, len(activation_list), 2, 252, 252, 252, 8, 8,
                            64, len(activation_list), 2, 252, 252, 252, 8, 8,
                            64, len(activation_list), 2, 252, 252, 252, 8, 8,
                            3,   # загоовок блока декодера 1
                            32, len(activation_list), 2, 252, 252, 252,
                            32, len(activation_list), 2, 252, 252, 252, 8, 8,
                            32, len(activation_list), 2, 252, 252, 252, 8, 8,
                            32, len(activation_list), 2, 252, 252, 252, 8, 8,
                            3,   # загоовок блока декодера 0
                            16, len(activation_list), 2, 252, 252, 252,
                            16, len(activation_list), 2, 252, 252, 252, 8, 8,
                            16, len(activation_list), 2, 252, 252, 252, 8, 8,
                            16, len(activation_list), 2, 252, 252, 252, 8, 8)


    def ProcessConvLastLayer(self, Neyrons, x, Gene, Str):
        if Gene[1] < 84:
            kernel_regularizer = None
            KernReg = ''
        elif Gene[1] < 168:
            kernel_regularizer = regularizers.L1(0.005 + (Gene[1] - 83)/500)
            KernReg = f', kernel_regularizer = regularizers.L1({0.005 + (Gene[1] - 83)/500})'
        else:
            kernel_regularizer = regularizers.L2(0.005 + (Gene[1] - 167)/500)
            KernReg = f', kernel_regularizer = regularizers.L1({0.005 + (Gene[1] - 167)/500})'

        if Gene[2] < 84:
            activity_regularizer = None
            ActReg = ''
        elif Gene[2] < 168:
            activity_regularizer = regularizers.L1(0.005 + (Gene[2] - 83)/500)
            ActReg = f', activity_regularizer = regularizers.L1({0.005 + (Gene[2] - 83)/500})'
        else:
            activity_regularizer = regularizers.L2(0.005 + (Gene[2] - 167)/500)
            ActReg = f', activity_regularizer = regularizers.L2({0.005 + (Gene[2] - 167)/500})'

        x = Conv2D(Neyrons, (Gene[3]+1, Gene[4]+1), activation=out_activation_list[int(Gene[0])], padding='same',
                       activity_regularizer = activity_regularizer, kernel_regularizer = kernel_regularizer)(x)
        Str.append(f'Conv2D( {Neyrons}, ({Gene[3]+1}, {Gene[4]+1}){KernReg}{ActReg}, activation={out_activation_list[int(Gene[0])]}')
        return x, Str

    def ProcessConv2DLayer(self, Input, Gene, Strides, Str, Neyrons = 0, ShowOut = -1):
        Strides = (Strides, Strides)

        if Gene[4] < 84:
            kernel_regularizer = None
            KernReg = ''
        elif Gene[4] < 168:
            kernel_regularizer = regularizers.L1(0.005 + (Gene[4] - 83)/500)
            KernReg = f', kernel_regularizer = regularizers.L1({0.005 + (Gene[4] - 83)/500})'
        else:
            kernel_regularizer = regularizers.L2(0.005 + (Gene[4] - 167)/500)
            KernReg = f', kernel_regularizer = regularizers.L1({0.005 + (Gene[4] - 167)/500})'

        if Gene[5] < 84:
            activity_regularizer = None
            ActReg = ''
        elif Gene[5] < 168:
            activity_regularizer = regularizers.L1(0.005 + (Gene[5] - 83)/500)
            ActReg = f', activity_regularizer = regularizers.L1({0.005 + (Gene[5] - 83)/500})'
        else:
            activity_regularizer = regularizers.L2(0.005 + (Gene[5] - 167)/500)
            ActReg = f', activity_regularizer = regularizers.L2({0.005 + (Gene[5] - 167)/500})'

        if Neyrons == 0:
            Neyrons = (Gene[0]+self.MinNeyrons) * self.KNeyrons if Debug == 0 else 1

        if ShowOut == -1:
            SO = ''
        else:
            SO = f'Out{ShowOut} = '

        if Gene[2] == 0 or Gene[3] < 84: # ActivationAfterNorm
            MustActivate = False
            x = Conv2D(Neyrons, (Gene[6]+1, Gene[7]+1), activation=activation_list[int(Gene[1])], padding='same',
                       activity_regularizer = activity_regularizer, kernel_regularizer = kernel_regularizer, strides = Strides)(Input)
            Str.append(f'{SO}Conv2D( {Neyrons}, ({Gene[6]+1}, {Gene[7]+1}){KernReg}{ActReg}, activation={activation_list[int(Gene[1])]}, strides = {Strides[0]})')
            S0 = ''
        else:
            MustActivate = True
            x = Conv2D(Neyrons, (Gene[6]+1, Gene[7]+1), padding='same',
                       activity_regularizer = activity_regularizer, kernel_regularizer = kernel_regularizer, strides = Strides)(Input)
            Str.append(f'Conv2D( {Neyrons}, ({Gene[6] + 1}, {Gene[7] + 1}){KernReg}{ActReg}, strides = {Strides[0]})')

        if Gene[3] >= 168:
            x = BatchNormalization()(x)

            if MustActivate:
                Str.append('BatchNormalization')
            else:
                Str.append(f'{SO}BatchNormalization')
        elif Gene[3] >= 84:
            x = SpatialDropout2D(0.0076 * (Gene[3] - 83))(x)
            if MustActivate:
                Str.append(f'SpatialDropout2D({0.0076 * (Gene[3] - 83)})')
            else:
                Str.append(f'{SO}SpatialDropout2D({0.0076 * (Gene[3] - 83)})')

        if MustActivate:
            x = Activation(activation_list[int(Gene[1])])(x)
            Str.append(f'{SO}Activation({activation_list[int(Gene[1])]})')

        return x, Str

    def ProcessConv2DTransposeLayer(self, x, Gene, Str):
        if Gene[4] == 0:
            kernel_regularizer = None
            KernReg = ''
        elif Gene[4] == 1:
            kernel_regularizer = regularizers.L1(0.01)
            KernReg = ';  KernReg L1'
        else:
            kernel_regularizer = regularizers.L2(0.01)
            KernReg = ';  KernReg L2'

        if Gene[5] == 0:
            activity_regularizer = None
            Act0 = ''
        elif Gene[5] == 1:
            activity_regularizer = regularizers.L1(0.01)
            Act0 = ';  ActReg=L1'
        else:
            activity_regularizer = regularizers.L2(0.01)
            Act0 = ';  ActReg=L2'

        if Gene[2] == 0 or Gene[3] < 84: # ActivationAfterNorm
            MustActivate = False
            x = Conv2DTranspose(((Gene[0]+self.MinNeyrons) * self.KNeyrons)  if Debug == 0 else 1 , (2, 2), strides=(2, 2), activation=activation_list[int(Gene[1])], padding='same')(x)
            Str.append(f'Conv2DTranspose({(Gene[0]+self.MinNeyrons) * self.KNeyrons}, (2,2), activation={activation_list[int(Gene[1])]}, strides = (2, 2))')
        else:
            MustActivate = True
            x = Conv2DTranspose(((Gene[0]+self.MinNeyrons) * self.KNeyrons)  if Debug == 0 else 1, (2, 2), strides=(2, 2), padding='same')(x)
            Str.append(f'Conv2DTranspose({(Gene[0] + self.MinNeyrons) * self.KNeyrons}, (2,2), strides = (2, 2))')
        if Gene[3] >= 168:
            x = BatchNormalization()(x)
            Str.append('BatchNormalization')
        elif Gene[3] >= 84:
            x = SpatialDropout2D(0.0076 * (Gene[3] - 83))(x)
            Str.append(f'SpatialDropout2D({0.0076 * (Gene[3] - 83)})')

        if MustActivate:
            x = Activation(activation_list[int(Gene[1])])(x)
            Str.append(f'Activation({activation_list[int(Gene[1])]})')

        return x, Str

    def ProcessInputType(self, InpType):
        self.InputType = InpType

    def ProcessUNetDowmBlock(self, x, Gene, ID, Connections, Str):
        Layers = Gene[0]+1

        Str.append(f'Слоев в блоке {Layers}')

        GenePos = BlockCaptionSize

        for i in range(Layers - 1):
            x, Str = self.ProcessConv2DLayer(x, Gene[GenePos:], Strides = 1, Str = Str)

            GenePos+= LayerSize

        # Последний слой обрабатываем по-особому: в нем могут быть Strides
        if Gene[1] == 2: # Пулинг за счет Strides.
            if Gene[2] == 1: # проброс
                Str.append("#Слой - проброс на энкодер:")
                out, Str = self.ProcessConv2DLayer(x, Gene[RefLayersDisp:], Strides=1, Str = Str, ShowOut=ID)
            else:
                out = x
                Str.append(f'out{ID} = x')
            Str.append("#Снижаем разрешение с помощью Strides:")
            x, Str = self.ProcessConv2DLayer(x, Gene[GenePos:], Strides=2, Str=Str)

        else: # обычный пулинг
            x, Str = self.ProcessConv2DLayer(x, Gene[GenePos:], Strides = 1, Str=Str) #conv слой

            if Gene[2] == 1: #проброс
                Str.append("#Слой - проброс на энкодер:")
                out, Str = self.ProcessConv2DLayer(x, Gene[RefLayersDisp:], Strides=1, Str = Str, ShowOut=ID)
            else:
                out = x
                Str.append(f'out{ID} = x')

            #pooling
            if Gene[1] == 0:
                x = MaxPooling2D()(x)
                Str.append(f'MaxPooling2D()')

            else:
                x = AveragePooling2D()(x)
                Str.append(f'AveragePooling2D()')

        Connections[ID] = out

        return x, Str

    def ProcessUNetUpBlock(self, x, Gene, Connections, Str):
        Layers = Gene[0]+1

        Str.append(f'Слоев в блоке {Layers}')

        GenePos = EncoderLayersStart

        for i in range(Layers):
            x, Str = self.ProcessConv2DLayer(x, Gene[GenePos:], Strides = 1, Str = Str)
            GenePos += LayerSize

        return x, Str

    def CreateNet(self, NetType, Gene, Inp):
        x = Inp
        Connections = {} # пробросы на другие слои. Представлены переменными на их выходах как значение, и номер слоя как ключ
                         # если номер слоя с минусом, то проброс идет на низходящую ветвь. Вверх пробросов не делаем.
        Str = [f'Уровней в сети {self.Layers}']                # Также невозможен проброс на следующий слой (нет смысла)
        if NetType == UNet:

            GenePos = LastLayerSize
            EncoderPos = GenePos + BlockSize * BlockCount
            for i in range(self.Layers):
                x, Str = self.ProcessUNetDowmBlock(x, Gene[GenePos:], i, Connections, Str = Str)
                GenePos+= BlockSize
                Str.append('----------')

            Str.append('--------------------------------------------')

            GenePos = EncoderPos
            for i in range(self.Layers):
                x, Str = self.ProcessConv2DTransposeLayer(x, Gene[GenePos + EncoderBlockCaptionSize:], Str)

                P = Connections.get(self.Layers - i - 1)

                if P != None:
                    x = concatenate([x, P])

                x, Str = self.ProcessUNetUpBlock(x, Gene[GenePos:], Connections, Str)
                GenePos += EncoderBlockSize

            x, Str = self.ProcessConvLastLayer(Inp.shape[-1], x, Gene, Str)

        return Model(Inp, x), Str


    # Собственно нешние методы

    def ProcessNet(self, Gene, Inp):
        if len(np.array(Inp).shape) == 1: # это размер данных
            input = Input(Inp)
        else:
            input = Input(Inp.shape[1:])


        self.Layers = Gene[0]+ 1

        Optimizer = optimizer_list[int(Gene[1])]
        self.BatchSz = (Gene[2] + 2)

        if self.HasInputType:
            SoftMaxFieldID = 4

            self.ProcessInputType(Gene[3])
        else:
            SoftMaxFieldID = 3

        Loss = 'mean_squared_error'
        if self.ResType == Classificator:
            if Gene[LastLayerPos] == 1: # на выходном слое активация softmax
                Loss  = 'categorical_crossentropy'

            HasNetFieldID = SoftMaxFieldID + 1
        else:
            if Gene[LastLayerPos] == 1:
                Loss = "binary_crossentropy"

            HasNetFieldID = SoftMaxFieldID + 1

        if self.NetType == Auto:
            self.NetType = Gene[HasNetFieldID]
            NextID = HasNetFieldID + 1
        else:
            NextID = HasNetFieldID

        # имеется ли поле типа входа в описании сет
        model, Str = self.CreateNet(self.NetType, Gene[LastLayerPos:], Inp=input)

        model.compile(optimizer=Optimizer, loss=Loss,metrics=[dice_coef])
        Str.append(f'optimizer={Optimizer}, loss={Loss}, Batch Size = {self.BatchSz}')

        return model, Str, Gene[NextID]

    def GenerateHromosom(self):
        Hr = np.random.randint(252, len(self.Mask)) % self.Mask

        # Предобработка
        Pos = FirstBlockPos

        for i in range(7): # Проходим по всем блокам
            Pr = Hr[Pos + 2]

            if Pr <= i or (Pr > 3 and Pr < i + 3):
                Hr[Pos + 2] = 0 # недопустимые значения просто обнуляем.

            Hr[Pos + 4]
            if Pr <= i or (Pr > 3 and Pr < i + 3):
                Hr[Pos + 4] = 0 # недопустимые значения просто обнуляем.

            Pos += BlockSize

        return Hr