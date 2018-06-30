from TrainingExample import *


class Seasonal_Q_Net():
    def __init__(self, Q_NetList):
        self.Q_NetList = Q_NetList
        Season_Num = len(Q_NetList)
        self.SeasonalTrainingExample = list()
        for season in range(Season_Num):
            self.SeasonalTrainingExample.append(list())

        self.EpochList = [2000, 2000, 5000, 3000, 2000, 2000, 2000, 2000]

    def CollectTrainingExample(self, season_count, new_example):
        self.SeasonalTrainingExample[season_count].append(new_example)

    def TrainSeasonalNetwork(self, season_count):
        ThisSeasonalExampleList = TrainingExampleList(self.SeasonalTrainingExample[season_count])
        ThisX_Q, ThisY_Q = ThisSeasonalExampleList.ReBuildAllMatrix()
        self.Q_NetList[season_count].fit(ThisX_Q, ThisY_Q, epochs=5000, batch_size=120)

    def SeasonalPredict(self, season_count, ThisInput):
        Predicted_Q = self.Q_NetList[season_count].predict(ThisInput.T, batch_size=None, verbose=0, steps=None)
        return Predicted_Q

