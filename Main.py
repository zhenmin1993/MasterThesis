from Battery import *
from PowerGrid import *
from Load import *
from CostFunction import *
#from ReadLoad import *
import numpy as np
import copy
import math
import random

#from GeneticAlgorithm import *
from TrainingExample import *
# from GridSearch import *
import matplotlib.pylab as plt
from SelectAction import *

######## Define the basic parameter of the network #########

House_num = 15

cos_phi = 0.95
phi = math.acos(cos_phi)

P_house_kW = 2
Q_house_kW = P_house_kW * math.tan(phi)
MyGrid = PowerGrid(House_num, P_house_kW, Q_house_kW)

P_max_onehouse_pu = P_house_kW / 1000
Q_max_onehouse_pu = Q_house_kW / 1000
S_max_onehouse_pu = math.sqrt(P_max_onehouse_pu * P_max_onehouse_pu + Q_max_onehouse_pu * Q_max_onehouse_pu)

TransformerRated = House_num * S_max_onehouse_pu * 0.8
TransformerRated_kW = TransformerRated * 1000
print('TransformerRated:', TransformerRated)
#################################################################


######## Define the parameter of the Batteries #########
P_batt_charge_pu = 1 / 1000
P_batt_discharge_pu = 1 / 1000
E_batt_max = 2 / 1000
RealBatteryList = list()

House_HasBattery = [0, 1, 2, 3, 4]
batt_number = len(House_HasBattery)
for i in range(House_num):
    if i in House_HasBattery:
        NewBattery = Battery(i, 0.1, E_batt_max, P_batt_charge_pu, P_batt_discharge_pu)
        RealBatteryList.append(NewBattery)
    else:
        RealBatteryList.append(0)

HouseBatteriesClass = BatteryList(RealBatteryList)
#################################################################


######## Generate load matrix #########


import xlrd


# route = '/Users/apple/Desktop/KU Leuven/Master Thesis/Ch3/load_profiles.xlsx'
def ReadLoadProfile(route, House_num):
    data = xlrd.open_workbook(route)

    table = data.sheets()[5]

    n_rows = table.nrows
    print('number of rows:', n_rows)  # 行数

    n_cols = table.ncols  # 列数

    tot_load = list()
    for i in range(1, n_rows):
        rowValues = table.row_values(i)[1:n_cols]  # 某一行数据
        # print(rowValues)
        tot_load = tot_load + rowValues

    tot_length = len(tot_load)
    loadArray = np.array(tot_load).reshape(1, tot_length)

    # print(load_matrix.shape)
    loadMatrix = np.tile(loadArray, (House_num)).reshape([House_num, tot_length])

    return loadMatrix


# print(ReadLoadProfile(route, 20).shape)


def GenDiffLoad(original_load_matrix):
    matrix_shape = original_load_matrix.shape
    max_load = np.max(original_load_matrix)
    min_load = np.min(original_load_matrix)
    np.random.seed(42)
    randLoad = np.random.random(size=matrix_shape) * (max_load - min_load) * 0.05

    NewLoad = np.add(original_load_matrix, randLoad)

    return NewLoad


route = '/Users/apple/Desktop/KU Leuven/Master Thesis/Ch3/load_profiles_Annual.xlsx'
table_no = 4
LoadMatrix = GenDiffLoad(ReadLoadProfile(route, House_num))

tot_hours = LoadMatrix.shape[1]
RealHouseLoadList = list()
for house in range(House_num):
    ThisLoad = Load(house, LoadMatrix[house, :].reshape([1, tot_hours]), cos_phi, P_max_onehouse_pu)
    RealHouseLoadList.append(ThisLoad)

OriginalLoad = copy.deepcopy(RealHouseLoadList)
OriginalLoadClass = LoadList(OriginalLoad)
HouseLoadListClass = LoadList(RealHouseLoadList)

# VirtualHouseLoadListClass = copy.deepcopy(HouseLoadListClass)

print(RealHouseLoadList[1].Shifted_ActiveLoad.shape)


def GetLoadFromLoadList(HouseLoadList, hour_count):
    House_num = len(HouseLoadList)
    ActiveLoadThisHour = np.array(np.zeros([House_num, 1]))
    ReactiveLoadThisHour = np.array(np.zeros([House_num, 1]))
    for house in range(House_num):
        ActiveLoadThisHour[house] = HouseLoadList[house].Shifted_ActiveLoad[0, hour_count]
        ReactiveLoadThisHour[house] = HouseLoadList[house].Shifted_ReactiveLoad[0, hour_count]

    return ActiveLoadThisHour, ReactiveLoadThisHour


def ShiftLoad(HouseLoadList, hour_count, S_insufficiency_Array):
    for load in HouseLoadList:
        load.LoadShift(S_insufficiency_Array, hour_count)


def RebuildLoadMatrix(ThisLoadList):
    ThisAllLoad = ThisLoadList.HouseLoadList
    House_num = len(ThisAllLoad)
    tot_hours = ThisAllLoad[0].Shifted_ActiveLoad.shape[1]
    ShiftedActive = np.array(np.zeros([House_num, tot_hours]))
    ShiftedReactive = np.array(np.zeros([House_num, tot_hours]))
    for house in range(House_num):
        ShiftedActive[house, :] = ThisAllLoad[house].Shifted_ActiveLoad[0, :]
        ShiftedReactive[house, :] = ThisAllLoad[house].Shifted_ReactiveLoad[0, :]
    return ShiftedActive, ShiftedReactive


ShiftedLoad = copy.deepcopy(LoadMatrix)

DecisionPeriod = 24
Day_num = int(tot_hours / DecisionPeriod)
###################################################################


######## Define the power supply sequence #########
Day_SupplyArray = np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0])
SupplyArray = np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0])
SupplyArray = SupplyArray.reshape([1, DecisionPeriod])
SupplyArray = np.tile(SupplyArray, (1, Day_num)).reshape([1, tot_hours])
#################################################################


######## Initialize the voltage array and power flow array #########
HouseVoltageArray = np.array(np.zeros([House_num, tot_hours]))
TransformerVoltageArray = np.array(np.zeros([1, tot_hours]))
TransformerFlowList = np.array(np.zeros([1, tot_hours]))


#################################################################


#######Function for running the model#############
def RunSimulation(hour, MyGrid, TransformerRated_kW, SupplyArray, HouseLoadList, BatteryList, Action):
    House_num = len(HouseLoadList)

    SoCList = list()

    S_insufficiency_Array = np.zeros([House_num, 1])

    HouseVoltage = 1
    TransformerVoltage = 1
    S_Trans = 0

    if SupplyArray[0, hour] == 1:
        ActiveLoadThisHour, ReactiveLoadThisHour = GetLoadFromLoadList(HouseLoadList, hour)

        results, S_Trans = MyGrid.HasSupply(ActiveLoadThisHour, ReactiveLoadThisHour, BatteryList, Action)

        if results['success'] == 1:
            print('Successful')

            HouseVoltage = results['bus'][2][7]
            TransformerVoltage = results['bus'][1][7]

        if results['success'] == 0:
            print('Unsuccessful')
            HouseVoltage = 0
            TransformerVoltage = -1
            S_Trans = -TransformerRated_kW * 2

        for house in range(House_num):
            # HouseVoltageArray[house,i] = HouseVoltage
            if BatteryList[house] != 0:
                # BatteryList[house].RealCharge(HouseVoltage)
                SoCList.append(BatteryList[house].SoC * 1000)

    if SupplyArray[0, hour] == 0:
        ActiveLoadThisHour, ReactiveLoadThisHour = GetLoadFromLoadList(HouseLoadList, hour)
        S_insufficiency_Array = MyGrid.NoSupply(ActiveLoadThisHour, ReactiveLoadThisHour, BatteryList, Action)

        for house in range(House_num):
            if BatteryList[house] != 0:
                SoCList.append(BatteryList[house].SoC * 1000)

        # HouseVoltage = 1
        # TransformerVoltage = 1
        # S_Trans = 0
    S_insufficiency_Array = np.multiply(S_insufficiency_Array, 1000)
    return HouseVoltage, TransformerVoltage, S_Trans, S_insufficiency_Array, SoCList


# print(RunSimulation(13, MyGrid ,SupplyArray ,RealHouseLoadList, RealBatteryList, 1))
# print(RunSimulation(11, MyGrid ,SupplyArray ,RealHouseLoadList, RealBatteryList, -1))
############################################################


###################################################################

####################Neural network using Keras####################
# from keras.models import Sequential

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
import keras

input_dimension = batt_number + 2 * House_num + 1 + batt_number
######## Build Model Neural Betwork ################
LearnedModel = Sequential()
LearnedModel.add(Dense(units=5, input_dim=input_dimension, activation='relu'))
LearnedModel.add(Dense(units=10, activation='relu'))
LearnedModel.add(Dense(units=10, activation='relu'))
LearnedModel.add(Dense(units=10, activation='relu'))
LearnedModel.add(Dense(units=2, activation='relu'))
EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
# Compile model
# optimizer = SGD(lr=self.best_learn_rate)
LearnedModel.compile(loss=keras.losses.mean_squared_error,
                     optimizer=keras.optimizers.SGD(lr=0.01))

######## Build Q-Network #############
Q_Net = Sequential()
Q_Net.add(Dense(units=20, input_dim=input_dimension, activation=keras.layers.advanced_activations.ELU(alpha=1.0)))
Q_Net.add(Dense(units=40, activation=keras.layers.advanced_activations.ELU(alpha=1.0)))
Q_Net.add(Dense(units=20, activation=keras.layers.advanced_activations.ELU(alpha=1.0)))
Q_Net.add(Dense(units=10, activation=keras.layers.advanced_activations.ELU(alpha=1.0)))
Q_Net.add(Dense(units=1, activation='sigmoid'))
EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
# Compile model
# optimizer = SGD(lr=self.best_learn_rate)
Q_Net.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.SGD(lr=0.01))

####################Define functions for grid search (Both Q_net and Model Net can use) ###################
learn_rate_space = [0.01, 0.02, 0.05]  #####
neurons_1_space = [10, 50, 100]
neurons_2_space = [10, 50, 100]
neurons_3_space = [10, 50, 100]


def create_Q_Net(learn_rate=0.01, neurons_1=5, neurons_2=5, neurons_3=5):
    # create model
    model = Sequential()
    model.add(Dense(neurons_1, input_dim=41, activation=keras.layers.advanced_activations.ELU(alpha=1.0)))
    model.add(Dense(neurons_2, activation=keras.layers.advanced_activations.ELU(alpha=1.0)))
    model.add(Dense(neurons_3, activation=keras.layers.advanced_activations.ELU(alpha=1.0)))
    model.add(Dense(1, activation='sigmoid'))
    EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    # Compile model
    optimizer = SGD(lr=learn_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    return model


def FindBestConfig_Q(X, Y, learn_rate_space, neurons_1_space, neurons_2_space, neurons_3_space):
    print(X.shape)
    print(Y.shape)
    seed = 7
    np.random.seed(seed)

    # create model
    model = KerasClassifier(build_fn=create_Q_Net, batch_size=24, epochs=1000, verbose=0)
    # define the grid search parameters

    param_grid = dict(learn_rate=learn_rate_space, neurons_1=neurons_1_space, neurons_2=neurons_2_space,
                      neurons_3=neurons_3_space)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

    grid_result = grid.fit(X, Y)
    print(X.shape)
    print(Y.shape)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    best_learn_rate = grid_result.best_params_['learn_rate']
    best_neurons_1 = grid_result.best_params_['neurons_1']
    best_neurons_2 = grid_result.best_params_['neurons_2']
    best_neurons_3 = grid_result.best_params_['neurons_3']

    return best_learn_rate, best_neurons_1, best_neurons_2, best_neurons_3


def CreateReal_Q_Net(X, Y, learn_rate_space, neurons_1_space, neurons_2_space, neurons_3_space):
    best_learn_rate, best_neurons_1, best_neurons_2, best_neurons_3 = FindBestConfig_Q(X, Y, learn_rate_space,
                                                                                       neurons_1_space, neurons_2_space,
                                                                                       neurons_3_space)
    Realmodel = Sequential()
    Realmodel.add(
        Dense(units=best_neurons_1, input_dim=41, activation=keras.layers.advanced_activations.ELU(alpha=1.0)))
    Realmodel.add(Dense(units=best_neurons_2, activation=keras.layers.advanced_activations.ELU(alpha=1.0)))
    Realmodel.add(Dense(units=best_neurons_3, activation=keras.layers.advanced_activations.ELU(alpha=1.0)))
    Realmodel.add(Dense(units=1, activation='sigmoid'))
    EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    # Compile model
    # optimizer = SGD(lr=self.best_learn_rate)
    Realmodel.compile(loss=keras.losses.mean_squared_error,
                      optimizer=keras.optimizers.SGD(lr=best_learn_rate))

    print('created')
    return Realmodel


#######################################################################


##################   Run    ######################
def ListMultiply(targetlist, number):
    NewList = list()
    for count in range(len(targetlist)):
        NewList.append(targetlist[count] * number)

    return NewList


def ReBuildInputVector(hour, SupplyArray, RealHouseLoadListClass, RealBatteryListClass, RealAction):
    # SoC * 2 + Load *2 *5 + Supply + action * 2
    ThisSupply = SupplyArray[0, hour]
    ThisSupply = np.array([ThisSupply]).reshape([1, 1])

    ThisLoad = RealHouseLoadListClass.GetLoadHour(hour)
    ThisLoad_kW = ListMultiply(ThisLoad, 1000)
    ThisLoad_kW = np.array(ThisLoad_kW).reshape(len(ThisLoad_kW), 1)

    ThisSoC = RealBatteryListClass.GetSoCList()
    ThisSoC_Percent = ListMultiply(ThisSoC, 1000 / 2)
    ThisSoC_Percent = np.array(ThisSoC_Percent).reshape(len(ThisSoC_Percent), 1)

    ThisAction = np.array(RealAction).reshape([len(RealAction), 1])
    InputVector = np.concatenate((ThisSoC_Percent, ThisLoad_kW, ThisSupply, ThisAction), axis=0)

    return InputVector


def ReBuildOutputVector(RealHouseVoltage, RealS_Trans):
    ###New： Voltage + S_trans

    ThisHouseVoltage = np.array([RealHouseVoltage]).reshape([1, 1])
    ThisS_Trans = np.array([RealS_Trans]).reshape([1, 1])

    OutputVector = np.concatenate((ThisHouseVoltage, ThisS_Trans), axis=0)

    return OutputVector


import pandas as pd

#################### S_Trans Related #######################################
S_TransDestruct = dict()

PredictedS_TransPercentHistory = list()
RealS_TransPercentHistory = list()

PredictedAverageS_TransPercentHistory = list()
RealAverageS_TransPercentHistory = list()
AverageS_TransDiffList = list()
S_TransDayExcess = list()
S_TransDayExcess.append(0)
S_TransPercentBox = pd.DataFrame()
S_TransPercentDaySeries = pd.Series([])
S_TransPercentPattern = list()
S_TransPercentPatternCount = list()
for i in range(24):
    S_TransPercentPattern.append(0)
    S_TransPercentPatternCount.append(0)


def WriteS_TransVoltagePattern(PatternList, PatternCountList, hour_in_day, new_value):
    PatternList[hour_in_day] = PatternList[hour_in_day] + new_value
    PatternCountList[hour_in_day] = PatternCountList[hour_in_day] + 1


################################################################

############## Voltage Related ################################
PredictedVoltageList = list()
RealVoltageList = list()
VoltageDayExcess_up = list()
VoltageDayExcess_down = list()
VoltageDayExcess_up.append(0)
VoltageDayExcess_down.append(0)
VoltagePattern = list()
VoltagePatternCount = list()
for i in range(24):
    VoltagePattern.append(0)
    VoltagePatternCount.append(0)

VoltageBox = pd.DataFrame()
VoltageDaySeries = pd.Series([])

PredictedAverageVoltageList = list()
RealAverageVoltageList = list()
AverageVoltageDiffList = list()
###########################################################


############## SoC Related ################################
RealSoCHistory = list()
for batt in range(batt_number):
    RealSoCHistory.append(list())


def WriteSoC(RealSoCHistory, SoCList):
    batt_number = len(SoCList)
    for batt in range(batt_number):
        RealSoCHistory[batt].append(SoCList[batt])


SoCPattern = list()
SoCPatternCount = list()
for batt in range(batt_number):
    SoCPattern.append(list())
    SoCPatternCount.append(list())
    for i in range(24):
        SoCPattern[batt].append(0)
        SoCPatternCount[batt].append(0)


def WriteSoCPattern(SoCPattern, SoCPatternCount, hour_in_day, SoCList):
    batt_number = len(SoCList)
    for batt in range(batt_number):
        SoCPattern[batt][hour_in_day] = SoCPattern[batt][hour_in_day] + SoCList[batt]
        SoCPatternCount[batt][hour_in_day] = SoCPatternCount[batt][hour_in_day] + 1


###########################################################


############## Insufficiency Related ################################
RealInsuffHistory = list()
tempInsuff24hList = list()


def WriteInsuff(InsufficiencyArray, RealInsufficiencyHistory, tempInsuff24hList):
    House_num = InsufficiencyArray.shape[0]
    for house in range(House_num):
        RealInsufficiencyHistory[house].append(InsufficiencyArray[house, 0])
        tempInsuff24hList[house] = tempInsuff24hList[house] + InsufficiencyArray[house, 0]


InsufficiencyPattern = list()
InsufficiencyPatternCount = list()

for house in range(House_num):
    InsufficiencyPattern.append(list())
    InsufficiencyPatternCount.append(list())
    for i in range(24):
        InsufficiencyPattern[house].append(0)
        InsufficiencyPatternCount[house].append(0)


def WriteInsufficiencyPattern(InsufficiencyPattern, InsufficiencyPatternCount, hour_in_day, InsufficiencyArray):
    House_num = len(InsufficiencyPattern)
    for house in range(House_num):
        InsufficiencyPattern[house][hour_in_day] = InsufficiencyPattern[house][hour_in_day] + InsufficiencyArray[
            house, 0]
        InsufficiencyPatternCount[house][hour_in_day] = InsufficiencyPatternCount[house][hour_in_day] + 1


AverageInsuffHistory = list()

###########################################################


##################### Load Related  ######################################
RealActiveLoadHistory = list()
CumulativeOrigLoad = list()
cumu_orig = list()
CumulativeShiftedLoad = list()
cumu_shifted = list()
###########################################################################


# Initialize all lists

for house in range(House_num):
    RealInsuffHistory.append(list())
    tempInsuff24hList.append(0)

    AverageInsuffHistory.append(list())
    RealActiveLoadHistory.append(list())

    CumulativeOrigLoad.append([0])
    # cumu_orig.append(0)
    CumulativeShiftedLoad.append([0])
    # cumu_shifted.append(0)
    print(RealInsuffHistory)

InitialtempInsuff24hList = copy.deepcopy(tempInsuff24hList)


def WriteAverageInsuff(AverageInsuffHistory, tempInsuff24hList):
    House_num = len(AverageInsuffHistory)
    for house in range(House_num):
        AverageInsuffHistory[house].append(tempInsuff24hList[house] / 24)


RealActiveLoadHistory_1 = list()
RealActiveLoadHistory_2 = list()
RealActiveLoadHistory_3 = list()
RealActiveLoadHistory_4 = list()
RealActiveLoadHistory_5 = list()


def WriteLoadList(CumulativeLoadList, ThisLoadHour):
    House_num = len(CumulativeLoadList)
    tot_hour_written = len(CumulativeLoadList[0])
    for house in range(House_num):
        lastload = CumulativeLoadList[house][tot_hour_written - 1]
        CumulativeLoadList[house].append(lastload + ThisLoadHour[2 * house])


##initialize temporary variables##
tempVoltage24h_real = 0
tempVoltage24h_predict = 0
tempTransformerPowerPercent24h_real = 0
tempTransformerPowerPercent24h_predict = 0
##################################


SupplyHistory = list()
TrainingExampleBuffer_Model = list()
TrainingExampleBuffer_Q = list()
RealQ_ValueList = list()
PredictedQ_ValueList = list()


def BuildVirtualInputVector(hour, SupplyArray, HouseBatteriesClass, HouseLoadListClass, RandomAction):
    # SoC * 2 + Load *2 *5 + Supply + action * 2
    ThisSupply = SupplyArray[0, hour]
    ThisSupply = np.array([ThisSupply]).reshape([1, 1])

    ThisLoad = HouseLoadListClass.GetLoadHour(hour)
    ThisLoad_kW = ListMultiply(ThisLoad, 1000)
    ThisLoad_kW = np.array(ThisLoad_kW).reshape(len(ThisLoad_kW), 1)

    ThisSoC = HouseBatteriesClass.GetSoCHour(hour)
    ThisSoC_Percent = ListMultiply(ThisSoC, 1000 / 2)
    ThisSoC_Percent = np.array(ThisSoC_Percent).reshape(len(ThisSoC_Percent), 1)

    ThisAction = np.array(RealAction).reshape([len(RandomAction), 1])
    InputVector = np.concatenate((ThisSoC_Percent, ThisLoad_kW, ThisSupply, ThisAction), axis=0)

    return InputVector


def ParseState(StateVector):
    ThisVoltage = StateVector[0].tolist()[0]
    ThisS_TransPercent = StateVector[0].tolist()[1]

    return ThisVoltage, ThisS_TransPercent


ThisSelectAction = SelectAction(batt_number)


def VirtualSimulation(selected_hour, VirtualSteps, SupplyArray, LearnedModel, ThisSelectAction, MySeasonalNetworks,
                      HouseBatteriesClass, HouseLoadListClass):
    VirtualHouseBatteriesClass = copy.deepcopy(HouseBatteriesClass)
    VirtualHouseLoadListClass = copy.deepcopy(HouseLoadListClass)
    Maximum_TotalReward = 0
    Total_Reward = 0
    #season_count = HourToSeason(selected_hour)
    season_count = 0
    Epsilon = 1
    DecayFactor = 0.8

    for step in range(VirtualSteps):
        Epsilon = Epsilon * DecayFactor
        VirtualHour = selected_hour + step
        ThisSupply = SupplyArray[0, VirtualHour]
        # RandomActionNumber = ActionSpace(len(HouseBatteriesClass.GetHouseTags()))
        # RandomActionList = RandomActionNumber.Number2Action()
        if ThisSupply == 1:
            ThisSelectAction.BuildState(VirtualHour, VirtualHouseBatteriesClass, VirtualHouseLoadListClass, SupplyArray)
            ThisQ_Net = MySeasonalNetworks.Q_NetList[season_count]
            random_num = random.random()
            ThisActionList = ThisSelectAction.FindMaxQValue(ThisQ_Net, random_num, Epsilon)

        if ThisSupply == 0:
            ThisActionList = [0, 0, 0, 0, 0]

        ThisInput = BuildVirtualInputVector(VirtualHour, SupplyArray, VirtualHouseBatteriesClass,
                                            VirtualHouseLoadListClass, ThisActionList)
        ThisLoad = VirtualHouseLoadListClass.GetLoadHour(VirtualHour)
        ModelPrediction = LearnedModel.predict(ThisInput.T, batch_size=None, verbose=0, steps=None)
        ThisVoltage, ThisS_TransPercent = ParseState(ModelPrediction)
        ThisInsufficiencyArray, NewSoC = VirtualHouseBatteriesClass.OperateBatteryList(ThisSupply, ThisVoltage,
                                                                                       ThisLoad, ThisActionList)
        VirtualHouseLoadListClass.LoadShift(ThisInsufficiencyArray, VirtualHour)
        ThisReward = RewardFunction(ThisVoltage, ThisS_TransPercent, sum(ThisInsufficiencyArray) * 1000)
        Total_Reward = Total_Reward + ThisReward * (0.9 ** step)
        Maximum_TotalReward = Maximum_TotalReward + 1 * (0.9 ** step)

        if step == 0:
            InitialStateAction = copy.deepcopy(ThisInput)

    ThisQ = Total_Reward / Maximum_TotalReward
    NewExample_Q = TrainingExample(InitialStateAction, np.array([ThisQ]))
    MySeasonalNetworks.CollectTrainingExample(season_count, NewExample_Q)
    # Q_Net.fit(X, Y, epochs=800, batch_size=48)


StepAhead = 48
Maximum_TotalReward = 0
for step in range(StepAhead):
    Maximum_TotalReward = Maximum_TotalReward + 1 * (0.9 ** step)

tempRewardList = list()
tempPredictedQList = list()
tempTrainingExampleList_Q = list()
SeasonalTrainingExampleList_Q = list()
Q_NetList = [Q_Net, Q_Net, Q_Net, Q_Net, Q_Net, Q_Net, Q_Net, Q_Net]

from SeasonalNetworks import *  # Spring, Spring->Summer,Summer, Summer->Fall,Fall,Fall->Winter,Winter,Winter->Spring

MySeasonalNetworks = Seasonal_Q_Net(Q_NetList)

DecayFactor = 0.9
Epsilon = 1
VirtualRounds = 240
Run_Time = tot_hours - 40
#Run_Time = 50
old_season = 0

season_count = 0

# Run_Time = 50
def HourToSeason(hour):
    day_count = int((hour + 1) / 24)
    season_count = int((day_count + 1) / 10)
    if season_count >= 8:
        season_count = season_count - 8 * int(season_count / 8)

    return season_count


for hour in range(Run_Time):

    print('This is hour : ', hour)

    day_count = int((hour + 1) / 24)

    #season_count = int((day_count + 1) / 10)
    #if season_count != old_season:
     #   old_season = copy.deepcopy(season_count)
      #  Epsilon = 1

    #if season_count >= 8:
     #   season_count = season_count - 8 * int(season_count / 8)
    # Planning = GA(StepAhead, LearnedModel, hour, HouseBatteriesClass, VirtualHouseLoadListClass, SupplyArray, TransformerRated)
    SupplyHistory.append(SupplyArray[0, hour])
    print('This Supply: ', SupplyArray[0, hour])

    HouseBatteriesClass = BatteryList(RealBatteryList)
    HouseLoadListClass = LoadList(RealHouseLoadList)

    print('SoC at hour :', hour, HouseBatteriesClass.GetSoCHour(hour))
    print('Load at this hour:', HouseLoadListClass.GetLoadHour(hour))

    hour_in_day = int(hour % 24)
    # print(hour_in_day)
    WriteSoCPattern(SoCPattern, SoCPatternCount, hour_in_day, HouseBatteriesClass.GetSoCList())
    # SoCPattern[hour_in_day] = SoCPattern[hour_in_day] + HouseBatteriesClass.GetSoCList()[0]
    # SoCPatternCount[hour_in_day] = SoCPatternCount[hour_in_day] + 1

    LoadHour = HouseLoadListClass.GetLoadHour(hour)
    LoadHour_kW = ListMultiply(LoadHour, 1000)

    RealActiveLoadHistory_1.append(LoadHour_kW[0])
    RealActiveLoadHistory_2.append(LoadHour_kW[2])

    WriteLoadList(CumulativeShiftedLoad, LoadHour_kW)

    OrigLoadHour = OriginalLoadClass.GetLoadHour(hour)
    OrigLoadHour_kW = ListMultiply(OrigLoadHour, 1000)

    WriteLoadList(CumulativeOrigLoad, OrigLoadHour_kW)

    # Run for 24 hours randomly to collect training data
    if hour >= 23:
        Epsilon = Epsilon * DecayFactor
        if SupplyArray[0, hour] == 1:
            ThisSelectAction.BuildState(hour, HouseBatteriesClass, HouseLoadListClass, SupplyArray)
            random_num = random.random()
            RealAction = ThisSelectAction.FindMaxQValue(MySeasonalNetworks.Q_NetList[season_count], random_num, Epsilon)
            print(RealAction)

        if SupplyArray[0, hour] == 0:
            RealAction = [0, 0, 0, 0, 0]

    if hour < 23:
        if SupplyArray[0, hour] == 0:
            RealAction = [0, 0, 0, 0, 0]

        if SupplyArray[0, hour] == 1:
            RealAction = [1, 1, 1, 1, 1]

    ThisInput = ReBuildInputVector(hour, SupplyArray, HouseLoadListClass, HouseBatteriesClass, RealAction)

    Predicted_Model = LearnedModel.predict(ThisInput.T, batch_size=None, verbose=0, steps=None)
    # Predicted_Q = Q_Net.predict(ThisInput.T, batch_size=None, verbose=0, steps=None)
    Predicted_Q = MySeasonalNetworks.SeasonalPredict(season_count, ThisInput)
    tempPredictedQList.append(Predicted_Q[0][0])
    # print('Predicted_Q_Value:', Predicted_Q)

    RealHouseVoltage, RealTransformerVoltage, RealS_Trans, RealS_insufficiency_Array, RealSoCList = RunSimulation(hour,
                                                                                                                  MyGrid,
                                                                                                                  TransformerRated_kW,
                                                                                                                  SupplyArray,
                                                                                                                  RealHouseLoadList,
                                                                                                                  RealBatteryList,
                                                                                                                  RealAction)

    RealSoCList_kW = ListMultiply(RealSoCList, 1000)

    RealS_TransPercent = RealS_Trans / TransformerRated_kW
    HouseLoadListClass.LoadShift(RealS_insufficiency_Array / 1000, hour)
    print('Insuff:', RealS_insufficiency_Array)
    ThisReward = RewardFunction(RealHouseVoltage, RealHouseVoltage, np.sum(RealS_insufficiency_Array))
    print('This Reward:', ThisReward)
    tempRewardList.append(ThisReward)

    if RealS_TransPercent > 1:
        S_TransDayExcess[day_count] = S_TransDayExcess[day_count] + 1
        if RealS_TransPercent > 1.7:
            S_TransDestruct[hour] = RealS_TransPercent

    if RealHouseVoltage > 1.05:
        VoltageDayExcess_up[day_count] = VoltageDayExcess_up[day_count] + 1

    if RealHouseVoltage < 0.95:
        VoltageDayExcess_down[day_count] = VoltageDayExcess_down[day_count] - 1

    ThisOutput_Model = ReBuildOutputVector(RealHouseVoltage, RealS_TransPercent)

    # LearnedModel.GetTrainingExamples(ThisInput, ThisOutput)

    # LearnedModel.Train()

    TrainingExampleBuffer_Model.append(TrainingExample(ThisInput, ThisOutput_Model))
    tempTrainingExampleList_Q.append(TrainingExample(ThisInput, np.array([0]).reshape([1, 1])))
    if len(tempRewardList) == StepAhead:
        ThisQ_Value = 0
        for step in range(StepAhead):
            ThisQ_Value = ThisQ_Value + tempRewardList[step] * (0.9 ** step)
        ThisQ_Value = ThisQ_Value / Maximum_TotalReward
        NewExample_Q = TrainingExample(tempTrainingExampleList_Q[0].InputStateAction,
                                       np.array([ThisQ_Value]).reshape([1, 1]))
        MySeasonalNetworks.CollectTrainingExample(season_count, NewExample_Q)
        PredictedQ_ValueList.append(tempPredictedQList[0])
        print('Predicted Q Value:', tempPredictedQList[0])
        if abs((tempPredictedQList[0] - ThisQ_Value) / ThisQ_Value) >= 0.05:
            Epsilon = 1
        tempTrainingExampleList_Q.pop(0)
        tempRewardList.pop(0)
        tempPredictedQList.pop(0)
        RealQ_ValueList.append(ThisQ_Value)

        print('Real Q Value:', ThisQ_Value)

    ThisTrainingExampleList_Model = TrainingExampleList(TrainingExampleBuffer_Model)
    # ThisTrainingExampleList_Q = TrainingExampleList(TrainingExampleBuffer_Q)

    X_Model, Y_Model = ThisTrainingExampleList_Model.ReBuildAllMatrix()
    # X_Q, Y_Q = ThisTrainingExampleList_Q.ReBuildAllMatrix()
    # print(X)

    if (hour + 1) % 24 == 0:

        VoltageBox[day_count - 1] = VoltageDaySeries
        VoltageDaySeries = pd.Series([])

        S_TransPercentBox[day_count - 1] = S_TransPercentDaySeries
        S_TransPercentDaySeries = pd.Series([])

        S_TransDayExcess.append(0)
        VoltageDayExcess_up.append(0)
        VoltageDayExcess_down.append(0)


        for i in range(VirtualRounds):
            selected_hour = random.randint(0, hour - 20)
            VirtualSimulation(selected_hour, StepAhead, SupplyArray, LearnedModel, ThisSelectAction,
                              MySeasonalNetworks, HouseBatteriesClass,
                              HouseLoadListClass)

        if (hour + 1) % 72 == 0:
            MySeasonalNetworks.TrainSeasonalNetwork(season_count)

        if (hour + 1) % (24 * 30) == 0:
            # Search1 = GridSearchFunc()
            print('Start Grid Search')
            # LearnedModel = CreateRealModel(X_Model, Y_Model)

            # Q_Net = CreateReal_Q_Net(X_Q, Y_Q, learn_rate_space, neurons_1_space, neurons_2_space, neurons_3_space)
            print('Grid Search complete')

            print('model created')

        LearnedModel.fit(X_Model, Y_Model, epochs=2000, batch_size=48)

        averageVoltage24h_real = tempVoltage24h_real / 24
        RealAverageVoltageList.append(averageVoltage24h_real)

        averageVoltage24h_predict = tempVoltage24h_predict / 24
        PredictedAverageVoltageList.append(averageVoltage24h_predict)

        AverageVoltageDiffList.append(averageVoltage24h_real - averageVoltage24h_predict)

        averageTransformerPowerPercent24h_real = tempTransformerPowerPercent24h_real / 24
        RealAverageS_TransPercentHistory.append(averageTransformerPowerPercent24h_real)

        averageTransformerPowerPercent24h_predict = tempTransformerPowerPercent24h_predict / 24
        PredictedAverageS_TransPercentHistory.append(averageTransformerPowerPercent24h_predict)

        AverageS_TransDiffList.append(
            averageTransformerPowerPercent24h_real - averageTransformerPowerPercent24h_predict)

        WriteAverageInsuff(AverageInsuffHistory, tempInsuff24hList)

        tempVoltage24h_real = 0
        tempVoltage24h_predict = 0
        tempTransformerPowerPercent24h_real = 0
        tempTransformerPowerPercent24h_predict = 0
        # tempInsuff24h_house1 = 0
        # tempInsuff24h_house2 = 0
        # tempInsuff24h_house3 = 0
        # tempInsuff24h_house4 = 0
        # tempInsuff24h_house5 = 0

        tempInsuff24hList = copy.deepcopy(InitialtempInsuff24hList)

    # Predicted = LearnedModel.predict(ThisInput.T, batch_size=None, verbose=0, steps=None)

    # Predicted = LearnedModel.Predict(ThisInput)
    Predicted_Model = Predicted_Model.T
    PredictedVoltageList.append(Predicted_Model[0, 0])
    tempVoltage24h_predict = tempVoltage24h_predict + Predicted_Model[0, 0]
    RealVoltageList.append(RealHouseVoltage)
    tempVoltage24h_real = tempVoltage24h_real + RealHouseVoltage
    WriteS_TransVoltagePattern(VoltagePattern, VoltagePatternCount, hour_in_day, RealHouseVoltage)

    ThisVoltageBias = RealHouseVoltage - Predicted_Model[0, 0]
    VoltageDaySeries = VoltageDaySeries.append(pd.Series([ThisVoltageBias]), ignore_index=True)

    # RealSoCHistory.append(RealSoCList[0])
    WriteSoC(RealSoCHistory, RealSoCList_kW)

    WriteInsuff(RealS_insufficiency_Array, RealInsuffHistory, tempInsuff24hList)
    WriteInsufficiencyPattern(InsufficiencyPattern, InsufficiencyPatternCount, hour_in_day, RealS_insufficiency_Array)

    PredictedS_TransPercentHistory.append(Predicted_Model[1, 0] * 100)
    tempTransformerPowerPercent24h_predict = tempTransformerPowerPercent24h_predict + Predicted_Model[1, 0] * 100
    RealS_TransPercentHistory.append(RealS_TransPercent * 100)
    tempTransformerPowerPercent24h_real = tempTransformerPowerPercent24h_real + RealS_TransPercent * 100
    WriteS_TransVoltagePattern(S_TransPercentPattern, S_TransPercentPatternCount, hour_in_day, RealS_TransPercent)

    ThisS_TransBias = RealS_TransPercent - Predicted_Model[1, 0]
    S_TransPercentDaySeries = S_TransPercentDaySeries.append(pd.Series([ThisS_TransBias * 100]), ignore_index=True)

for batt in range(len(SoCPattern)):
    for hour in range(len(SoCPattern[0])):
        SoCPattern[batt][hour] = SoCPattern[batt][hour] / SoCPatternCount[batt][hour]

for hour in range(24):
    VoltagePattern[hour] = VoltagePattern[hour] / VoltagePatternCount[hour]
    S_TransPercentPattern[hour] = S_TransPercentPattern[hour] / S_TransPercentPatternCount[hour]

for house in range(House_num):
    for hour in range(24):
        InsufficiencyPattern[house][hour] = InsufficiencyPattern[house][hour] / InsufficiencyPatternCount[house][hour]

################# Build Insuficiency Box Plot############################
InsufficiencyPatternBox_HasBattery = pd.DataFrame()
InsufficiencyPatternDaySeries_HasBattery = pd.Series([])
InsufficiencyPatternBox_NoBattery = pd.DataFrame()
InsufficiencyPatternDaySeries_NoBattery = pd.Series([])

AverageInsufficiencyPatternBox_HasBattery = pd.DataFrame()
AverageInsufficiencyPatternDaySeries_HasBattery = pd.Series([])
AverageInsufficiencyPatternBox_NoBattery = pd.DataFrame()
AverageInsufficiencyPatternDaySeries_NoBattery = pd.Series([])

for hour in range(24):
    hasbatt_list = list()
    nobatt_list = list()
    for house in range(House_num):
        if house < batt_number:
            hasbatt_list.append(InsufficiencyPattern[house][hour])

        if house >= batt_number:
            nobatt_list.append(InsufficiencyPattern[house][hour])

    InsufficiencyPatternBox_HasBattery[hour] = pd.Series(hasbatt_list)
    InsufficiencyPatternBox_NoBattery[hour] = pd.Series(nobatt_list)

for day in range(len(AverageInsuffHistory[0])):
    hasbatt_average = list()
    nobatt_average = list()
    for house in range(House_num):
        if house < batt_number:
            hasbatt_average.append(AverageInsuffHistory[house][day])

        if house >= batt_number:
            nobatt_average.append(AverageInsuffHistory[house][day])

    AverageInsufficiencyPatternBox_HasBattery[day] = pd.Series(hasbatt_average)
    AverageInsufficiencyPatternBox_NoBattery[day] = pd.Series(nobatt_average)

#########################################################################


# ThisTrainingExampleList = TrainingExampleList(TrainingExampleBuffer)
TotalTrainingExampleList_Model = TrainingExampleList(TrainingExampleBuffer_Model)
TotalTrainingExampleList_Q = TrainingExampleList(TrainingExampleBuffer_Q)
# global X
# global Y
X_Model, Y_Model = TotalTrainingExampleList_Model.ReBuildAllMatrix()
Z_Model = np.concatenate((X_Model, Y_Model), axis=1)

# X_Q, Y_Q = TotalTrainingExampleList_Q.ReBuildAllMatrix()
# Z_Q = np.concatenate((X_Q,Y_Q), axis = 1)
Folder_Path = '/Users/apple/PycharmProjects/MGRL_Paper/ModelAssistedRL/TransformerProtectionCase/Lol_1/'

np.savetxt(Folder_Path + 'TrainingExample_Model.csv', Z_Model)
# np.savetxt(Folder_Path+ 'TrainingExample_Q.csv', Z_Q)
# print(X)
OriginalActiveMatrix, OriginalReactiveMatrix = RebuildLoadMatrix(OriginalLoadClass)
np.savetxt(Folder_Path + 'OriginalActiveLoadMatrix.csv', OriginalActiveMatrix[:, 0:Run_Time])
np.savetxt(Folder_Path + 'OriginalReactiveLoadMatrix.csv', OriginalReactiveMatrix[:, 0:Run_Time])
ShiftedActiveMatrix, ShiftedReactiveMatrix = RebuildLoadMatrix(HouseLoadListClass)
np.savetxt(Folder_Path + 'ShiftedActiveLoadMatrix.csv', ShiftedActiveMatrix[:, 0:Run_Time])
np.savetxt(Folder_Path + 'ShiftedReactiveLoadMatrix.csv', ShiftedReactiveMatrix[:, 0:Run_Time])

VoltageMatrix = np.array([PredictedVoltageList, RealVoltageList])
np.savetxt(Folder_Path + 'VoltageMatrix.csv', VoltageMatrix)
AverageVoltageMatrix = np.array([PredictedAverageVoltageList, RealAverageVoltageList])
np.savetxt(Folder_Path + 'AverageVoltageMatrix.csv', AverageVoltageMatrix)
np.savetxt(Folder_Path + 'VoltagePattern.csv', np.array(VoltagePattern))

TransformerPowerMatrix = np.array([PredictedS_TransPercentHistory, RealS_TransPercentHistory])
np.savetxt(Folder_Path + 'TransformerPowerMatrix.csv', TransformerPowerMatrix)
AverageTransformerPowerMatrix = np.array([PredictedAverageS_TransPercentHistory, RealAverageS_TransPercentHistory])
np.savetxt(Folder_Path + 'AverageTransformerPowerMatrix.csv', AverageTransformerPowerMatrix)
np.savetxt(Folder_Path + 'S_TransPercentPattern.csv', np.array(S_TransPercentPattern))

SoCMatrix = np.array(RealSoCHistory)
np.savetxt(Folder_Path + 'SoCMatrix.csv', SoCMatrix)
SoCPatternMatrix = np.array(SoCPattern)
np.savetxt(Folder_Path + 'SoCPatternMatrix.csv', SoCPatternMatrix)

InsufficiencyMatrix = np.array(RealInsuffHistory)
np.savetxt(Folder_Path + 'InsufficiencyMatrix.csv', InsufficiencyMatrix)
AverageInsufficiencyMatrix = np.array(AverageInsuffHistory)
np.savetxt(Folder_Path + 'AverageInsufficiencyMatrix.csv', AverageInsufficiencyMatrix)
InsufficiencyPatternMatrix = np.array(InsufficiencyPattern)
np.savetxt(Folder_Path + 'InsufficiencyPatternMatrix.csv', InsufficiencyPatternMatrix)

Q_ValueMatrix = np.array([PredictedQ_ValueList, RealQ_ValueList])
np.savetxt(Folder_Path + 'Q_ValueMatrix.csv', Q_ValueMatrix)

print(S_TransDestruct)

# LearnedModel.fit(X, Y , epochs=10, batch_size=32)
