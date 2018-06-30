import copy
import numpy as np

class Battery():
    def __init__(self, house_tag,initial_SoC, Capacity, Rated_Charge, Max_Discharge):
        self.house_tag = house_tag
        self.initial_SoC = initial_SoC * Capacity
        self.SoC = self.initial_SoC
        self.Capacity = Capacity
        self.Rated_Charge = Rated_Charge
        self.Max_Discharge = Max_Discharge
        self.SoCHour = list()
        self.SoCHour.append(self.SoC)
        self.Discharge_eff = 0.9
        self.Charge_eff = 0.9

    def Action(self, action_name, load):
        if action_name == 'Charge':
            return self.PreCharge()

        if action_name == 'Idling':
            return self.Idling()

        if action_name == 'DisCharge':
            return self.DisCharge(load)

        else:
            raise ValueError('There is an action unknown')

    def PreCharge(self):
        Old_SoC = copy.deepcopy(self.SoC)
        Pre_SoC = self.SoC + self.Rated_Charge * self.Charge_eff
        Pre_SoC = min(Pre_SoC, self.Capacity)
        pre_power = Pre_SoC - Old_SoC
        return pre_power/self.Charge_eff
        #return 0
    def RealCharge(self, real_voltage):
        self.SoC = self.SoC + self.Rated_Charge * real_voltage * real_voltage *self.Charge_eff
        self.SoC = min(self.SoC, self.Capacity)
        self.SoCHour.append(self.SoC)

    def Idling(self):
        self.SoCHour.append(self.SoC)
        return 0

    def DisCharge(self,load):
        Old_SoC = copy.deepcopy(self.SoC)
        self.SoC = self.SoC - min(load/self.Discharge_eff,self.Max_Discharge)
        self.SoC = max(self.SoC, 0)
        real_discharge = Old_SoC - self.SoC
        self.SoCHour.append(self.SoC)
        #print(real_discharge)
        #print(load)
        return real_discharge
        #return 0


class BatteryList():
    def __init__(self, Batteries):
        self.Batteries = Batteries

    def GetSoCList(self):
        SoCList = list()
        for battery in self.Batteries:
            if battery != 0:
                SoCList.append(battery.SoC)

        return SoCList

    def GetSoCHour(self, hour_selected):
        SoCHourList = list()
        for battery in self.Batteries:
            if battery != 0:
                SoCHourList.append(battery.SoCHour[hour_selected])

        return SoCHourList

    def GetHouseTags(self):
        HouseHasBattery = list()
        for battery in self.Batteries:
            if battery != 0:
                HouseHasBattery.append(battery.house_tag)

        return HouseHasBattery

    def OperateBatteryList(self, Supply ,real_voltage, load_list, action_list):
        batt_count = 0
        tot_house_count = 0
        InsufficiencyArray = np.zeros([len(self.Batteries), 1])
        for battery in self.Batteries:
            load = load_list[tot_house_count]
            if Supply == 1:

                if battery != 0:
                    action = action_list[batt_count]

                    batt_count = batt_count + 1

                    if action == 1:
                        battery.RealCharge(real_voltage)

                    if action == 0:
                        battery.Idling()

                InsufficiencyArray[tot_house_count,0] = 0

            if Supply == 0:

                if battery != 0:
                    action = action_list[batt_count]

                    batt_count = batt_count + 1

                    if action == 1:
                        battery.Idling()
                        InsufficiencyArray[tot_house_count, 0] = load

                    if action == 0:
                        real_discharge = battery.DisCharge(load)
                        InsufficiencyArray[tot_house_count, 0] = load - real_discharge * battery.Discharge_eff

                if battery == 0:
                    InsufficiencyArray[tot_house_count,0] = load

            tot_house_count = tot_house_count + 1

        SoCList = self.GetSoCList()
        return InsufficiencyArray, SoCList