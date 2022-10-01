import math
from BaseGenetic import TBaseGenetic
import numpy as np
import random
from NetFromHromosom import TNNHromosom, BinClassification
from  Settings import *
print(f'UNetGenetic v0.9   Debug = {Debug}')
from LocalSettings import *
from keras.callbacks import Callback

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

class TNetGenetic(TBaseGenetic):
    def __init__(self, InputX, InputY, TestX, TestY, Verbose, NetHromosom):
        self.NetHromosom = NetHromosom
        TBaseGenetic.__init__(self, HromosomLen = len(self.NetHromosom.Mask), FixedGroupsLeft=0, StopFlag=0, PopulationSize = 100)

        self.InputX = InputX
        self.InputY = InputY
        self.TestX = TestX
        self.TestY = TestY

        self.Verbose = Verbose

        self.epoches = 5

        self.StartLoadPaths =[] # список путей, из которых при старте будет попытка прочитать хромосомы


    def TryLoad(self, First = True):

    
        ModelList =  []
        ModelResList = []
        for Path in self.StartLoadPaths:
            Models = glob.glob(Path + "* res *.h5")
            for Model in Models:

                IDPos = Model.find(" ")
                ResPos = Model.find(" res ")
                ResEndPos = Model.find(".")
                ID = int(Model[IDPos+1:ResPos])
                Res = float(Model[ResPos+5:ResEndPos])

                ModelList.append({"ID" : ID, "ModelPath" : Path + Model, "NumpyPath" : f'{ID}.npy'})
                ModelResList.append(Res)

        ModelResList = np.array(ModelResList)
        Ind = ModelResList.sort()

        #[-self.StartPopulationSize if First else PopulationSize:]

        for I in Ind:
            ModelObj = ModelList[I]
            Hr = np.load(ModelObj["NumpyPath"])

            if len(Hr) < len(self.NetHromosom):
                self.Hromosoms[I] = np.concatenate([Hr[:]])
            else:
                self.Hromosoms[I] = Hr

        self.HromosomRatingValues[I] = list(ModelResList[Ind])

        return len(Ind)

    def TestHromosom(self, Hr, HrID):
        Net, Str, InpSz = self.NetHromosom.ProcessNet(Hr, (self.InputX))
        InpSz = 473 + 86*InpSz
        Str.append(f'Размер входного пакета {InpSz}')
        NegInpSz = -InpSz
        if self.Verbose>0:
            for s in Str:
                print(s)
            Net.summary()
        #Надо вычислять лучший
        CrCallback = THromosomCallback()

        try:
            Split = 100/InpSz

            if Debug == 0:

                Res = Net.fit(self.InputX[NegInpSz:], self.InputY[NegInpSz:], batch_size = 16, validation_split = Split, epochs = self.epoches,
                         verbose = self.Verbose, callbacks = [CrCallback])

                if Res[0] == None:
                    raise('None в метриках')

                Net.set_weights(CrCallback.BestW)
                Res = Net.evaluate(self.TestX, self.TestY, verbose=self.Verbose)

            elif Debug == 1:
                Net.fit(self.InputX[NegInpSz:], self.InputY[NegInpSz:], batch_size=16, epochs=self.epoches,validation_split = Split,
                         verbose=self.Verbose, callbacks=[CrCallback])
                Net.set_weights(CrCallback.BestW)
                Res = Net.evaluate(self.TestX, self.TestY, verbose=self.Verbose)
            else:
                Res = (0, random.random())


        except Exception as e:
            print(f'Сбой хромосомы: {e}')
            return None

        Str.insert(0, str(Res[1]))

        Net.save(f'{CrPath}{LI} {HrID} res {Res[1]}.h5')
        print(f'Сохранено {CrPath} {LI} {HrID} res {Res[1]}.h5')
        with open(f'{CrPath}{LI} {HrID}.txt', "w") as file:
            for S in Str:
                file.write(S+'\n')

        np.save(f'{CrPath}{LI} {HrID}.npy', Hr)

        if self.Verbose > 0:
            print(Res)

        return Res

    def InitNewHromosom(self, Res, GetNewID=True):
        TBaseGenetic.InitNewHromosom(self, Res, GetNewID)

        Res %= self.NetHromosom.Mask

class TUNetGenetic(TNetGenetic):
    def __init__(self, InputX, InputY, TestX, TestY, Verbose, MaxLevels):
        TNetGenetic.__init__(self, InputX, InputY, TestX, TestY, Verbose, TNNHromosom(ResType = BinClassification, MaxLevels = MaxLevels))

        self.MaxLevels = MaxLevels
