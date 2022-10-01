import numpy
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, BatchNormalization, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, \
    ZeroPadding2D, GlobalAveragePooling2D, concatenate
from keras.layers import add, Flatten, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.callbacks import TensorBoard, Callback
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from keras.applications import efficientnet_v2
from pathlib import Path
import glob, os
from PIL import Image
import pickle
from tensorflow.keras.utils import to_categorical
from keras.applications import efficientnet_v2, efficientnet, resnet
from keras import regularizers
import gc
import random
import time
import shutil

Epoches = 100

Colab = True
try:
    from google.colab import drive
except:
    Colab = False

if Colab:
    from google.colab import drive

    # Подключаем Google drive
    drive.mount('/content/drive')
    CrPath = "/content/drive/MyDrive/"
else:
    Acer = not os.path.exists("E:/w/Diplom/")
    CrPath = "C:/w/Diplom/" if Acer else "E:/w/Diplom/"

activation_list = ['relu', 'elu', 'selu', 'tanh', 'sigmoid']
optimizer_list = ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam']

Henetic = CrPath + 'Henetic/'
ResPath = Henetic + 'Res.npy'
StatePath = Henetic + "state.npy"
BestPath = Henetic + "best"
BestNetPath = Henetic + "BestNet"
BestWeightsPath = Henetic + "BestWeights"

class TGenetic:
    '''
        StopFlag: 0 - never stop
                  1 - stop after n populations
                  2 - stop when Metris is More then MetricLimit
    '''
    def __init__(self, StopFlag = 0, TheBestListSize = 10, HrCnt0 = 50, PopulationSize = 50):
        self.HrCnt0 = HrCnt0
        self.HrCnt1 = HrCnt1

        # The list of the best result for any time
        self.TheBestListSize = TheBestListSize # the number of storing best hromosoms
        self.TheBestList = [None]*TheBestListSize # The list of the copy of best Hromosoms
        self.TheBestValues = np.zeros(TheBestListSize)  # The list of the best results

        # Alive hromosoms and there raitings
        self.Hromosoms = [] #
        self.HromosomRatingValues = []
        self.RaitingRaitingIndexes = []  # the raiting of the hromosoms (indexes)

        self.StopFlag = StopFlag
        self.Metric = 0
        self.MetricLimit = 1
        self.GenerationsLimit = 100
        self.Generation = 0

        self.PMutation = 0.2 # probability of the mutation for any individuals. Or, if >=1, the number of individuals that will die in each generation
        self.PMultiMutation = 0.1 # the probability of an any bit to be mutated, in the case of the mutation has occurred
        self PDeath = 10 # probability of the death. Or, if >=1, the number of individuals that will die in each generation


    def GenerateHromosom(self):
        return 0

    def TestHromosom(self, Hr):
        return random.random()

    # returns True if the end of evalution. The result is Ready
    def Stop(self):
        if self.StopFlag == 1:
            return self.Generation >= self.GenerationsLimit
        elif self.StopFlag == 2:
            return self.Metric >= self.MetricLimit
        else:
            return False #never stop

    def TestHromosoms(self):
        self.HromosomRatingValues = [TestHromosom(H) for H in Hromosoms]

    def MutateHromosom(self, Hr, MutList):
        # process the mutation of the single hromosom.
        # Hr - the reference to the mutated hromosom. MutList - numpy array, the same length as the Hromosom, has a
        # random values from 0 to 1. can be use to can be used to determine which bits of a chromosome must be mutated
        # mutate.
        # returns the mutated Hromosom
        Mutations = GenerateHromosom()
        return np.where(MutList <= self.PMultiMutation, Mutations, Hr)

    def StoreTheBest():
        # store the best hromosoms
        HrCnt = len(self.HromosomRatingValues)
        CrRaiting = np.argsort(np.append(self.HromosomRatingValues, self.TheBestValues))[-self.TheBestListSize:]

        for Ind in reversed(CrRaiting):
            BestPos = 0

            LastBestlist = self.TheBestList.copy()
            if Ind >= HrCnt:
                self.TheBestList[BestPos] = self.TheBestList[Ind - HrCnt]
                self.TheBestValues[BestPos] = self.TheBestValues[Ind - HrCnt]
            else:
                self.TheBestList[BestPos] = self.Hromosom[Ind]
                self.TheBestValues[BestPos] = self.HromosomRatingValues
            BestPos+= 1

    def Mutations():
        N = self.PMutation if self.PMutation < 1 else round(random.gauss(N, 1))
        L = len(Hromosoms)
        Muts = np.random.randint(0, L) # the list of the mutated hromosoms

        for Hr in Muts:
            MutList = np.random.rand(L)

            Hromosoms[Hr] = MutateHromosom(Hromosoms[Hr], MutList)

    def Reproductions(self): # для ускорения вероятность зависит от места
        CrRaiting = np.argsort( self.HromosomRatingValues, self.TheBestValues )


        '''
        Sum = 0
        for Ind, RaitingValue in enumerate(CrRating):
            Sum += RaitingValue

            if Ind == 0:
                continue

            CrRaiting[Ind] = 
         '''

    def Start(self):
        # Generate Hromosoms List
        self.Hromosoms = [GenerateHromosom() for i in range(HrCnt0)]

        while not Stop():
            TestHromosoms()
            StoreTheBest()
            Mutations()
            Reproductions()
            Deaths()

def CreateNetFromHromosom(net, Verbose):
        for i, Gene in enumerate(net):

            if i == 0:
                makeFirstNormalization = Gene[0]  # Делаем ли нормализацию в начале
                NetType = Gene[
                    1]  # 0 - ResNet, 1- efficientnet_v2 2 - efficientnet_B7 3 - ResNet обученная на наших данных

                if NetType == 0:
                    NetType = 1
                elif NetType == 3:
                    NetType = 2

                LearningRate = 0.0001 + 0.0002 * Gene[2]  # 0,02..0,0001
                # BatchSz = int(Gene[3])  # BatchSz 1 - 1024

                Optimizer = optimizer_list[int(Gene[5])]  #

                InputSz = 1280

                if NetType == 0:
                    File = "ResNet"

                elif NetType == 1:
                    File = "EfficientNetV2L"

                elif NetType == 2:
                    File = "EfficientNetB7"
                    InputSz = 2560

                elif NetType == 3:
                    File = "MyResNet"

                SoftMax = Gene[
                    4]  # Кодирует softMax или sigmoid. Если sigmoid, то используем один выходной нейрон, и sigmoid.
                # Иначе 2 и softmax.
                if SoftMax == 1:
                    Loss = 'categorical_crossentropy'
                else:
                    Loss = "binary_crossentropy"

                x = Input(shape=(InputSz,))

                input = x

            else:
                if Gene[0] == 0:
                    break  # больше слоев не будет

                if Gene[1] == 0:
                    kernel_regularizer = None
                    KernReg = ''
                elif Gene[1] == 1:
                    kernel_regularizer = regularizers.L1(0.01)
                    KernReg = ';  KernReg L1'
                else:
                    kernel_regularizer = regularizers.L2(0.01)
                    KernReg = ';  KernReg L2'

                if Gene[2] == 0:
                    activity_regularizer = None
                    Act0 = ''
                elif Gene[2] == 1:
                    activity_regularizer = regularizers.L1(0.01)
                    Act0 = ';  ActReg=L1'
                else:
                    activity_regularizer = regularizers.L2(0.01)
                    Act0 = ';  ActReg=L2'

                x = Dense(int(Gene[0]), activation=activation_list[int(Gene[3])],
                          activity_regularizer=activity_regularizer,
                          kernel_regularizer=kernel_regularizer)(x)

                print('Dense', Gene[0], '; activation', activation_list[int(Gene[3])], Act0, KernReg)

                if Gene[4] == 1:
                    x = BatchNormalization()(x)

                    print('BatchNormalization')
                elif Gene[4] > 0:
                    x = Dropout(0.0001 * (Gene[4] - 1))(x)

                    print('Dropout(', 0.0001 * (Gene[4] - 1), ')')

        if SoftMax == 1:
            x = Dense(2, activation="softmax")(x)

            print('Выход: Dense(2, activation="softmax")')
        else:
            x = Dense(1, activation="sigmoid")(x)

            print('Выход: Dense(1, activation="sigmoid")')

        model = Model(inputs=input, outputs=x)

        model.compile(optimizer=Optimizer, loss=Loss, metrics=['accuracy'])

        # K.set_value(model.optimizer.learning_rate, LearningRate)

        if Verbose:
            model.summary()
            print('Net type', File, '   optimizer=', Optimizer, '   loss=', Loss)

        return model, File, SoftMax

class TLearningCallback(Callback):
    def __init__(self, Path, FileIndex, State):
        self.Path = Path
        if State == None:
            self.State = [0, 0] # лучшее значение, его позиция
        else:
            self.State = State

        self.FileIndex = str(FileIndex)

    def on_epoch_end(self, epoch, logs=None):
        CR = logs['val_accuracy']
        if CR > self.State[0]:
            self.State[0] = CR
            self.State[1] = epoch

            print(' Сохранен лучший, acc=', CR, 'эпоха', epoch + 1)

            self.model.save_weights(self.Path + 'WLBest_' + self.FileIndex + '.h5')

        #self.model.save_weights(self.Path + 'WL_' + self.FileIndex + '.h5')
        #np.save(self.Path + 'WLState' + self.FileIndex + '.h5', self.State)

def LearnTheBest(FileIndex = '000', BatchSz=128, Verbose=True):
    global f

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=0.0001)

    Net = np.load(CrPath + 'Henetic/BestNet' + FileIndex + '.dat.npy')  # формат (10,5,5)
    Results = np.load(CrPath + 'Henetic/best' + FileIndex + '.dat.npy')

    if Path(CrPath + 'Henetic/LearningState.npy').exists():
        LearningState = np.load(CrPath + 'Henetic/LearningState.npy')
    else:
        LearningState = np.array( (0,0) ) # хромосома/эпоха

    LearningState[0] = 6

    for i, Hr in enumerate(Net):
            if i < LearningState[0]:
                print(i)
                continue

            model, File, SoftMax = CreateNetFromHromosom(Hr, Verbose)

            XTrain = numpy.load(CrPath + "Henetic/" + File + "/X.npy")
            YTrain = numpy.load(CrPath + "Henetic/" + File + "/Y.npy")
            XTest = numpy.load(CrPath + "Henetic/" + File + "/XTest.npy")
            YTest = numpy.load(CrPath + "Henetic/" + File + "/YTest.npy")

            if SoftMax == 0:
                YTrain = YTrain[:, 0]
                YTest = YTest[:, 0]

            Nm = BestWeightsPath + str(i) + '_' + str(int(FileIndex)) + '.npy'
            print(Nm)
            if Path(Nm).exists():
                    CrWeights = np.load(Nm, allow_pickle=True)
                    try:
                        model.set_weights(CrWeights)
                        model.save_weights(CrPath + 'WLBest_' + str(i) + '.h5')
                        print('Создан ', CrPath + 'WLBest_' + str(i) + '.h5')
                        State = [Results[i]/1.02, 0]
                    except:
                        print('Сбой веса', i)
                        State = [0, 0]

                    print('Загружены веса, ', i)
            else:
                    print('Отсутствуют веса, ', i)

            #if i == 0:
            #    continue


            CheckPoint = TLearningCallback(CrPath + "Henetic/", FileIndex + '_' + str(i), State)

            H = model.fit(XTrain, YTrain,
                  batch_size=BatchSz,
                  epochs=Epoches,
                  verbose=0,#1 if Verbose else 0,
                  callbacks=(reduce_lr, TLearningCallback(CrPath + "Henetic/", str(i), State)),
                  validation_data=(XTest, YTest))

            VA = H.history["val_accuracy"]
            ACC = H.history["accuracy"]

            BestEpoch = np.argmax(VA)
            print('Итог во время эволюции', Results[i], ', итог после дообучения', VA[BestEpoch],
                  '. Лучшая эпоха', BestEpoch)
            print()

            plt.plot(VA, label='Доля верных ответов на обучающем наборе')
            plt.plot(ACC, label='Доля верных ответов на тренирующем наборе')

            plt.xlabel('Эпоха обучения')
            plt.ylabel('Доля верных ответов')
            plt.legend()

            plt.show()

            gc.collect()

LearnTheBest(FileIndex = '038')