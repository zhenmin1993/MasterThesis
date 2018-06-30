import numpy as np

class TrainingExample():
    def __init__(self, InputStateAction, OutputState):
        self.InputStateAction = InputStateAction
        self.OutputState = OutputState


class TrainingExampleList():
    def __init__(self, TrainingExampleList):
        self.TrainingExampleList = TrainingExampleList

    def ReBuildAllMatrix(self):
        InputDimension = self.TrainingExampleList[0].InputStateAction.shape[0]
        OutputDimesion = self.TrainingExampleList[0].OutputState.shape[0]
        X_Train = np.zeros([len(self.TrainingExampleList), InputDimension ])
        Y_Train = np.zeros([len(self.TrainingExampleList), OutputDimesion ])
        for count in range(len(self.TrainingExampleList)):
            X_Train[count, :] = self.TrainingExampleList[count].InputStateAction.reshape([1,InputDimension])
            Y_Train[count, :] = self.TrainingExampleList[count].OutputState.reshape([1,OutputDimesion])

        return X_Train, Y_Train

#####The state has to include the hour information to enable time sequence planning afterwards######
class State():
    def __init__(self, StateAction, hour):
        ThisHour = np.array(hour).reshape([1,1])
        self.State =np.concatenate((ThisHour, StateAction[0, 1:]),axis = 0)
