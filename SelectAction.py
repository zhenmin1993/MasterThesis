import random
import numpy as np
import copy


class SelectAction():
    def __init__(self, Battery_num):
        self.Battery_num = Battery_num
        self.OneTimeAction = 2 ** self.Battery_num
        self.AllActionNumber = list()
        for scalar in range(self.OneTimeAction):
            self.AllActionNumber.append(scalar + 1)

        # self.Action = random.randint(1,self.OneTimeAction)

    def Number2Action(self, original_number):
        conv = bin(original_number - 1)
        action_list = list(conv[2:])
        # new_list = list()
        deficit = self.Battery_num - len(action_list)
        if deficit > 0:
            for i in range(deficit):
                action_list.insert(0, 0)
        # print('length of action list', len(action_list))
        for i in range(len(action_list)):
            action_list[i] = int(action_list[i])
        # print('length of action list',len(action_list))
        return action_list

    def ListMultiply(self, targetlist, number):
        NewList = list()
        for count in range(len(targetlist)):
            NewList.append(targetlist[count] * number)

        return NewList

    def BuildState(self, hour, HouseBatteriesClass, HouseLoadListClass, SupplyArray):
        ThisSupply = SupplyArray[0, hour]
        ThisSupply = np.array([ThisSupply]).reshape([1, 1])

        ThisLoad = HouseLoadListClass.GetLoadHour(hour)
        ThisLoad_kW = self.ListMultiply(ThisLoad, 1000)
        ThisLoad_kW = np.array(ThisLoad_kW).reshape(len(ThisLoad_kW), 1)

        ThisSoC = HouseBatteriesClass.GetSoCHour(hour)
        # print('SoCHour:',hour,ThisSoC)
        ThisSoC_Percent = self.ListMultiply(ThisSoC, 1000 / 2)
        ThisSoC_Percent = np.array(ThisSoC_Percent).reshape(len(ThisSoC_Percent), 1)

        self.ThisState = np.concatenate((ThisSoC_Percent, ThisLoad_kW, ThisSupply), axis=0)

        # return ThisState

    def FindMaxQValue(self, Q_Net, random_num, Epsilon):
        maxQ = 0
        minQ = 5500
        maxAction = self.Number2Action(self.AllActionNumber[0])
        minAction = self.Number2Action(self.AllActionNumber[0])
        # randomAction = self.Number2Action(random.randint(1,self.OneTimeAction))

        if random_num < Epsilon:
            maxAction = self.Number2Action(random.randint(1, self.OneTimeAction))

        if random_num >= Epsilon:
            for action_number in self.AllActionNumber:
                ThisActionList = self.Number2Action(action_number)
                ThisAction = np.array(ThisActionList).reshape([self.Battery_num, 1])
                # print('State,Action:',self.ThisState,ThisAction)
                ThisInput = np.concatenate((self.ThisState, ThisAction), axis=0)
                This_Q = Q_Net.predict(ThisInput.T, batch_size=None, verbose=0, steps=None)
                if This_Q > maxQ:
                    maxQ = copy.deepcopy(This_Q)
                    maxAction = copy.deepcopy(ThisActionList)

        return maxAction