import numpy as np


#############This function is the cost function of the neural network
def Calculate_Cost(Y_hat, Y):
    ExampleNum = Y.shape[1]
    #return - (np.dot(np.log(Y_hat), Y.T)+np.dot(np.log(1-Y_hat), 1-Y.T))
    return 1/2* (np.dot(Y_hat-Y, Y_hat.T-Y.T)) #+ lambda_reg/2/ExampleNum

#Y_hat = np.random.rand(1,100)
#Y = np.random.rand(1,100)*0.5
#print(Calculate_Cost(Y_hat, Y))


############This function is the reward function of different states###########
def RewardFunction(voltage, S_TransPercent, lost_load):
    reward = 300
    cost = 0
    VoLL = 1

    if  S_TransPercent > 1.7 and S_TransPercent <= 2:
        OverLoadCostPerKVA = 100 * (S_TransPercent ** 2) - 320 * S_TransPercent + 272
        reward = reward - OverLoadCostPerKVA * 25.26 - lost_load * VoLL

    if S_TransPercent > 1 and S_TransPercent <= 1.7:
        reward = reward - S_TransPercent * 10 * 25.26 - lost_load * VoLL
        #cost = 3.38 * (S_TransPercent ** 13.756) + lost_load * VoLL

    if S_TransPercent <= 1 and S_TransPercent >= 0:
        reward = reward - lost_load * VoLL
        #cost = lost_load * VoLL

    if S_TransPercent > 2 or S_TransPercent < 0:
        reward = 0
        #cost = 5500

    if reward < 0:
        reward = 0

    return reward/300