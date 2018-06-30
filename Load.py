import math
import numpy as np

class Load():
    def __init__(self, house_tag, ActiveLoadArray, cos_phi, P_max_onehouse_pu):
        self.cos_phi = cos_phi
        phi = math.acos(self.cos_phi)
        self.sin_phi = math.sqrt(1- cos_phi**2)
        self.house_tag = int(house_tag)
        self.tot_hours = ActiveLoadArray.shape[1]
        Q_max_onehouse_pu = P_max_onehouse_pu * math.tan(phi)
        self.Original_ActiveLoad = ActiveLoadArray * P_max_onehouse_pu
        self.Original_ReactiveLoad = ActiveLoadArray * Q_max_onehouse_pu
        self.Shifted_ActiveLoad = ActiveLoadArray * P_max_onehouse_pu
        self.Shifted_ReactiveLoad = ActiveLoadArray * Q_max_onehouse_pu

    def LoadShift(self, insufficiencyArray, hour_count):

        insufficiency = insufficiencyArray[self.house_tag,0]
        if insufficiency > 0:
            if hour_count <= self.tot_hours - 2:
                self.Shifted_ActiveLoad[0,hour_count + 1] = self.Shifted_ActiveLoad[0, hour_count + 1] + insufficiency * self.cos_phi *0.3
                self.Shifted_ReactiveLoad[0, hour_count + 1] = self.Shifted_ReactiveLoad[ 0,hour_count + 1] + insufficiency * self.sin_phi *0.3


class LoadList():
    def __init__(self, HouseLoadList):
        self.HouseLoadList = HouseLoadList

    def GetLoadHour(self, hour):
        LoadatHour = list()
        for load in self.HouseLoadList:
            LoadatHour.append(load.Shifted_ActiveLoad[0,hour])
            LoadatHour.append(load.Shifted_ReactiveLoad[0,hour])

        return LoadatHour

    def LoadShift(self, insufficiencyArray, hour):
        for load in self.HouseLoadList:
            load.LoadShift(insufficiencyArray, hour)



# -*- coding: utf-8 -*-
