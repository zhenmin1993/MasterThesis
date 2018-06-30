from pypower.api import ppoption, runpf

import math


class PowerGrid():

    def __init__(self, House_num, P_house_kW, Q_house_kW):
        self.House_num = House_num
        self.ppc = CaseThesis(House_num)
        self.ppopt = ppoption(PF_ALG=1)

        P_max_onehouse_pu = P_house_kW / 1000
        Q_max_onehouse_pu = Q_house_kW / 1000
        S_max_onehouse_pu = math.sqrt(P_max_onehouse_pu * P_max_onehouse_pu + Q_max_onehouse_pu * Q_max_onehouse_pu)
        TransformerRated = House_num * S_max_onehouse_pu * 0.8

        self.ppc['branch'][0, 5] = TransformerRated
        self.ppc['branch'][0, 6] = TransformerRated
        self.ppc['branch'][0, 7] = TransformerRated

        for j in range(House_num):
            self.ppc['branch'][j + 1, 5] = S_max_onehouse_pu * 1.1
            self.ppc['branch'][j + 1, 6] = S_max_onehouse_pu * 1.1
            self.ppc['branch'][j + 1, 7] = S_max_onehouse_pu * 1.1

        #print('ppc here')
        #print(self.ppc['branch'])


    def HasSupply(self, Active_ShiftedLoad, Reactive_ShiftedLoad, BatteryList, Action):
        #House_num = Active_ShiftedLoad.shape[0]
        batt_count = 0
        for house in range(self.House_num):
            Pload = Active_ShiftedLoad[house]
            Qload = Reactive_ShiftedLoad[house]
            Sload = math.sqrt(Pload * Pload + Qload * Qload)
            this_real_power = 0

            if BatteryList[house] != 0 and Action[batt_count] == 1:
                this_real_power = BatteryList[batt_count].PreCharge()

            if BatteryList[house] != 0 and Action[batt_count] == 0:
                this_real_power = BatteryList[batt_count].Idling()

            if BatteryList[house] != 0:
                batt_count = batt_count + 1


            self.ppc['bus'][ house + 2, 2] = Pload + this_real_power
            self.ppc['bus'][ house + 2, 3] = Qload
            #print('House', house,'Load:', Pload)

        results, s = runpf(self.ppc, self.ppopt)

        P_Trans = results['branch'][0][13] * 1000
        Q_Trans = results['branch'][0][14] * 1000
        S_Trans = math.sqrt(P_Trans*P_Trans+Q_Trans*Q_Trans)
        HouseVoltage = results['bus'][2][7]

        if results['success'] == 1:
            for house in range(self.House_num):
                if BatteryList[house] != 0 and Action[house] == 1:
                    BatteryList[house].RealCharge(HouseVoltage)
        #print('results',results['branch'][0])
        print('This S_Trans:', S_Trans)

        return results, S_Trans



    def NoSupply(self, Active_ShiftedLoad, Reactive_ShiftedLoad, BatteryList, Action):
        #House_num = Active_ShiftedLoad.shape[0]
        S_insufficiency_Array = np.array(np.zeros([self.House_num,1]))
        batt_count = 0
        for house in range(self.House_num):
            Pload = Active_ShiftedLoad[house]
            Qload = Reactive_ShiftedLoad[house]
            Sload = math.sqrt(Pload * Pload + Qload * Qload)

            #if BatteryList[house] == 0 :
                #this_real_discharge = 0
            #else:
            this_real_discharge = 0
            if BatteryList[house] != 0 and Action[batt_count] == 0:
                this_real_discharge = BatteryList[house].DisCharge(Sload)

            if BatteryList[house] != 0 and Action[batt_count] == 1:
                this_real_power = BatteryList[batt_count].Idling()

            if BatteryList[house] != 0:
                batt_count = batt_count + 1


            S_insufficiency = Sload - this_real_discharge*0.9
            S_insufficiency_Array[house] = S_insufficiency
        S_insufficiency_Array = S_insufficiency_Array.reshape([self.House_num,1])
        return S_insufficiency_Array


# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Power flow data for 9 bus, 3 generator case.
"""

from numpy import array
import numpy as np


def CaseThesis(House_num):
    """Power flow data for 9 bus, 3 generator case.
    Please see L{caseformat} for details on the case file format.
    Based on data from Joe H. Chow's book, p. 70.
    @return: Power flow data for 9 bus, 3 generator case.
    """
    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 1.0

    ## bus data
    Pload = list()
    Qload = list()
    total_load = 0
    one_active = 2/1000
    one_reactive = 0.3/1000
    one_S = math.sqrt(one_active*one_active+one_reactive*one_reactive)
    for i in range(House_num):
        Pload.append(one_active)
        Qload.append(one_reactive)
        total_load = total_load + one_S
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = array([
        [1, 3, 0,          0,     0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9],
        [2, 1, 0,          0,     0, 0, 1, 1, 0, 0.22, 2, 1.1, 0.9]]).reshape([2,13])

    ppc["branch"] = array([1, 2, 0, 5, 0, total_load, total_load, total_load, 0, 0, 1, -360, 360]).reshape([1,13])

    for house in range(House_num):
        one_bus = np.array([house+3, 1, Pload[house], Qload[house], 0, 0, 1, 1, 0, 0.22, 2, 1.1, 0.9]).reshape([1,13])
        one_branch = np.array([2, house+3, 0.05,  0.2, 0,     one_S, one_S, one_S, 0, 0, 1, -360, 360]).reshape([1,13])
        ppc["bus"] = np.append(ppc["bus"],one_bus,axis = 0)
        ppc["branch"] = np.append(ppc["branch"],one_branch,axis = 0)




    #ppc["bus"] = array([
    #    [1, 3, 0,          0,     0, 0, 1, 1, 0, 0.4, 1, 1.1, 0.9],
    #    [2, 1, 0,          0,     0, 0, 1, 1, 0, 0.22, 2, 1.1, 0.9],
    #    [3, 1, Pload[0], Qload[0], 0, 0, 1, 1, 0, 0.22, 2, 1.1, 0.9],
     #   [4, 1, Pload[1], Qload[1], 0, 0, 1, 1, 0, 0.22, 2, 1.1, 0.9],
    #    [5, 1, Pload[2], Qload[2], 0, 0, 1, 1, 0, 0.22, 2, 1.1, 0.9],
    #])

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = array([
        [0, 0,   0, 300, -300, 1, math.inf, 1, math.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #[2, 163, 0, 300, -300, 1, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #[3, 85,  0, 300, -300, 1, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    #ppc["branch"] = array([
    #    [1, 2, 0,     5, 0,       total_load, total_load, total_load, 0, 0, 1, -360, 360],
    #    [2, 3, 0.39,  1.5, 0,     one_S, one_S, one_S, 0, 0, 1, -360, 360],
    #    [2, 4, 0.39,  1.5, 0,     one_S, one_S, one_S, 0, 0, 1, -360, 360],
    #    [2, 5, 0.39,  1.5, 0,     one_S, one_S, one_S, 0, 0, 1, -360, 360],
    #])

    ##-----  OPF Data  -----##
    ## area data
    # area refbus
    ppc["areas"] = array([
        [1,5]
    ])

    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = array([
        [2, 1500, 0, 3, 0.11,   5,   150],
        [2, 2000, 0, 3, 0.085,  1.2, 600],
        [2, 3000, 0, 3, 0.1225, 1,   335]
    ])

    return ppc