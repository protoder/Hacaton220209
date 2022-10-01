import numpy
import os

import numpy as np
import random

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

class TBaseGenetic:
    '''
        StopFlag: 0 - never stop
                  1 - stop after n populations
                  2 - stop when Metric is More or Less then MetricLimit

        GenGroupSize - для удобства, гены можно разбить на группы. Например, относящиеся к одному слою.
                  При этом хромосому можно прорешейпить [Nbit/GenGroupSize, GenGroupSize]
                  Понятно, что количество ген в хромосоме должно делиться на GenGroupSize

        FixedGroupsLeft - сколько групп слева не могут быть разбиты кроссинговером
    '''
    def __init__(self, HromosomLen, GenGroupSize = 1, FixedGroupsLeft = 0, StopFlag = 0, TheBestListSize = 10, StartPopulationSize = 50, PopulationSize = 50):
        self.StartPopulationSize = StartPopulationSize # стартовое количество хромосом
        self.PopulationSize = PopulationSize # постоянное количество хромосом


        # The list of the best result for any time
        self.TheBestListSize = TheBestListSize # the number of storing best hromosoms
        self.TheBestList = [None]*TheBestListSize # The list of the copy of best Hromosoms
        self.TheBestValues = np.zeros(TheBestListSize)  # The list of the best results

        # Alive hromosoms and there Ratings
        self.Hromosoms = [0]*PopulationSize #
        self.HromosomRatingValues = []
        self.ArgRating = []

        self.StopFlag = StopFlag
        self.InverseMetric = True  # метрика убывает, т.е. оптимизация на убывание

        if self.StopFlag == 2 and self.InverseMetric:
            self.Metric = 10000000
        else:
            self.Metric = 0

        self.FirstStep = True
        self.MetricLimit = 1
        self.GenerationsLimit = 100
        self.Generation = 0

        self.PMutation = 0.2 # probability of the mutation for any individuals. Or, if >=1, the number of individuals that will die in each generation
        self.PMultiMutation = 0.1 # the probability of an any bit to be mutated, in the case of the mutation has occurred
        self.PDeath = 0.2 # probability of the death. Or, if >=1, the number of individuals that will die in each generation
        self.PCrossingover = 0.5

        self.GenGroupSize = GenGroupSize
        self.HromosomLen = HromosomLen  # В хромосому добавляем два флага. 1 - признак измененности. 2 - ссылка на доп. данные

        self.FixedGroupsLeft = FixedGroupsLeft + 3

        self.ReportStep = 1
        self.CrHromosomID = 0

    # Методы, которые скорее всего придется перекрывать в своей реализации
    def TestHromosom(self, Hr):
        return random.random()


    # Методы придется перекрывать при нелинейной структуре хромосом (ссылки на дополнительные данные и пр.)
    def TryLoad(self):
        return 0

    def Save(self):
        pass

    def InitNewHromosom(self, Res, GetNewID = True):
        Res[0] =  0 # Признак, что хромосома измерена.

        if GetNewID:
            Res[1] = self.CrHromosomID
            self.CrHromosomID+= 1

    def BeforeDelHromosoms(self, P):
        pass

    def GenerateHromosom(self, GetNewID = True):
        Res =  np.random.randint(252, size = self.HromosomLen)
        self.InitNewHromosom(Res, GetNewID)
        return Res

    def MutateHromosom(self, Hr, MutList):
        # process the mutation of the single hromosom.
        # Hr - the reference to the mutated hromosom. MutList - numpy array, the same length as the Hromosom, has a
        # random values from 0 to 1. can be use to can be used to determine which bits of a chromosome must be mutated
        # mutate.
        # returns the mutated Hromosom

        Mutations = self.GenerateHromosom()
        return np.where(MutList <= self.PMultiMutation, Mutations, Hr)


    def HromosomsPairing(self, P0, P1):
        if random.random() > self.PCrossingover:
            Sel = np.random.rand(self.HromosomLen) > 0.5
            Res = np.where(Sel, P0, P1)
        else:
            CrosingoverPos = random.randint(self.FixedGroupsLeft, self.HromosomLen / self.GenGroupSize - 1) * self.GenGroupSize
            Res =  np.concatenate([P0[:CrosingoverPos], P1[CrosingoverPos:]])

        self.InitNewHromosom(Res)

        return Res
    #  скорее всего, остальные методы в перекрытой верии будут неизменны



    # returns True if the end of evalution. The result is Ready
    def Stop(self):
        if self.StopFlag == 1:
            return self.Generation >= self.GenerationsLimit
        elif self.StopFlag == 2:
            if self.InverseMetric:
                return self.Metric <= self.MetricLimit
            else:
                return self.Metric >= self.MetricLimit
        else:
            return False #never stop

    def TestHromosoms(self):

        RV = [0]*self.HromosomsCnt

        for i, H in enumerate(self.Hromosoms):
            if H[0] == 1:
                RV[i] = self.HromosomRatingValues[i]
            else:
                while True:
                    R = self.TestHromosom(H[2:], H[1])

                    if R != None:
                        RV[i] = R
                        break

                self.Hromosoms[i][0] = 1


        self.HromosomRatingValues = RV

        self.ArgRating = np.argsort(RV)
        self.Metric = RV[self.ArgRating[0]]

    def StoreTheBest(self):

        # store the best hromosoms
        HrCnt = len(self.HromosomRatingValues)

        if self.InverseMetric:
            if self.FirstStep:
                CrRating = self.ArgRating
                self.FirstStep = False
            else:
                CrRating = np.argsort(np.append(self.HromosomRatingValues, self.TheBestValues))
        else:
            CrRating = np.argsort(np.append(self.HromosomRatingValues, self.TheBestValues))[::-1]

        BestPos = 0

        InList = set()

        for Ind in CrRating:

            #LastBestlist = self.TheBestList.copy()
            if Ind >= HrCnt:
                Hr = self.TheBestList[Ind - HrCnt]

                if Hr[1] in InList:
                    continue

                Value = self.TheBestValues[Ind - HrCnt]

            else:
                Hr = self.Hromosoms[Ind]

                if Hr[1] in InList:
                    continue

                Value = self.HromosomRatingValues[Ind]

            InList.add(Hr[1])

            self.TheBestList[BestPos] = Hr
            self.TheBestValues[BestPos] = Value

            if BestPos == 0:
                self.Metric = Value

            BestPos+= 1

            if BestPos == self.TheBestListSize:
                return


    def Mutations(self):
        L = self.HromosomLen
        N = self.PMutation if self.PMutation > 1 else round(random.gauss(self.HromosomsCnt * self.PMutation, 1))

        Muts = np.random.randint(0, self.HromosomsCnt, N) # the list of the mutated hromosoms

        for Hr in Muts:
            MutList = np.random.rand(L)
            MutList[0], MutList[1] = (0, 0) # берем все из мутации
            CrHr = self.Hromosoms[Hr]
            self.Hromosoms[Hr] = self.MutateHromosom(CrHr, MutList)


    def Deaths(self):
        MustDie = self.PDeath if self.PDeath > 1 else round(random.gauss(self.HromosomsCnt * self.PDeath, 1))

        if self.InverseMetric:
            P = [self.ArgRating[int(x)] for x in np.random.triangular(0, self.HromosomsCnt, self.HromosomsCnt, MustDie)]
        else:
            P = [self.ArgRating[int(x)] for x in np.random.triangular(0, 0, self.HromosomsCnt, MustDie)]

        self.BeforeDelHromosoms(P)
        self.Hromosoms = list(np.delete(self.Hromosoms, P, 0))
        self.HromosomRatingValues = list(np.delete(self.HromosomRatingValues, P, 0))


    def Reproductions(self):
        Cnt = self.HromosomsCnt

        Childs = self.PopulationSize - Cnt # сколько нужно детей

        Rating = np.argsort(self.HromosomRatingValues)

        if self.InverseMetric:
            P = np.random.triangular(0, 0, Cnt, 2*Childs).reshape((Childs, 2))
        else:
            P = np.random.triangular(0, Cnt, Cnt, 2*Childs).reshape((Childs, 2))

        AddList = [0]*Childs
        for i, CrP in enumerate(P):
            P0 = int(CrP[0])
            P1 = int(CrP[1])

            while (P0 == P1):
                P1 = int(np.random.triangular(0, Cnt, Cnt))

            AddList[i] = self.HromosomsPairing(self.Hromosoms[Rating[P0]], self.Hromosoms[Rating[P1]])

        self.Hromosoms.extend(AddList)

        '''
        Sum = 0
        for i, P in enumerate(zip(P0, P1)):
            CrP0 = int(P[0])
            CrP1 = int(P[1])

            while CrP0 == CrP1:
                Cr

            for iR, R in enumerate(Ratings):
                Sum = Sum + R

                if Sum >= P0:
                    Parent0 = iR
                    break

            for iR, R in enumerate(Ratings):
                Sum = Sum + R

                if Sum >= P1:
                    Parent1 = iR
                    break

            Hromosoms[i + self.HromosomsCnt] = HromosomsPairing(Hromosoms[P0], Hromosoms[P1])
            '''
    def Report(self, G, M):
        print(f'Поколение {G:5} : {M}', end='\r')

    def TryLoad(self):
        if False:
            self.FirstStep = False

        return False

    def Start(self):
        # Generate Hromosoms List

        NeedNew = self.StartPopulationSize - self.TryLoad()

        if NeedNew > 0:
            self.Hromosoms = [self.GenerateHromosom() for i in range(NeedNew)]

        Hear = 0
        VasReplacing = False


        while not self.Stop():
            LastMetric = self.Metric

            self.TestHromosoms()
            self.StoreTheBest()


            Hear += 1
            if self.Generation % self.ReportStep == 0 or LastMetric != self.Metric:
                self.Report(self.Generation, self.Metric)

                if LastMetric != self.Metric:
                    Hear = 0 # Признак, что на этом шаге были изменения
                    VasReplacing = False

            if Hear == 100 and not VasReplacing:
                if self.InverseMetric:
                    From = None
                    To = -self.TheBestListSize
                else:
                    From = self.TheBestListSize
                    To = None

                    #self.Hromosoms = list(np.array(self.Hromosoms)[self.ArgRating[self.TheBestListSize:]])

                Slide = self.ArgRating[From:To]
                self.Hromosoms = list(np.array(self.Hromosoms)[Slide])
                self.Hromosoms.extend(self.TheBestList)

                self.HromosomRatingValues = list(np.array(self.HromosomRatingValues)[Slide])
                self.HromosomRatingValues.extend(self.TheBestValues)

                VasReplacing = True
            else:
                self.Deaths()
                self.Reproductions()

                self.Mutations()

            self.Generation += 1

            self.Save() # сохраняем список хромосом, BestList и BestValues
    @property
    def HromosomsCnt(self):
        return len(self.Hromosoms)

