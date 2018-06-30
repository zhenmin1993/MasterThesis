
import random

class ActionSpace():
    def __init__(self, Battery_num):
        self.Battery_num = Battery_num
        self.OneTimeAction = 2 ** self.Battery_num
        self.Action = random.randint(1,self.OneTimeAction)

    def Number2Action(self):
        conv = bin(self.Action-1)
        action_list = list(conv[2:])
        # new_list = list()
        deficit = self.Battery_num - len(action_list)
        if deficit > 0:
            for i in range(deficit):
                action_list.insert(0, 0)
        #print('length of action list', len(action_list))
        for i in range(len(action_list)):
            action_list[i] = int(action_list[i])
        #print('length of action list',len(action_list))
        return action_list

    def Mutate(self):
        self.Action = random.randint(1,self.OneTimeAction)


class ActionSpaceList():
    def __init__(self, ActionList):
        self.ActionList = ActionList


    def GetAllActions(self):
        NewActionList = list()
        for action in self.ActionList:
            NewActionList.append(action.Number2Action())

        return NewActionList

