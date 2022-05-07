###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import pandas as pd
import numpy as np
import math
import random
import time

def Euclidean_distance(input_data, test_data):
    d = np.zeros([len(input_data)],dtype=float)
    for i in range(len(input_data)):
        r = 0
        for j in range(len(input_data.columns) - 1):
            r = r + pow(float(input_data.iloc[i,j]) - float(test_data.iloc[:,j]),2)
        d[i] = math.sqrt(r)
    return d

def knn(input_data,distances,n):
    label = 0
    idx = np.argpartition(distances, n)
    k_min_index = idx[:n]
    positive = 0
    negative = 0
    mean = 0
    for i in k_min_index:
        mean = mean + float(input_data.iloc[i,5])
    mean = mean/k_min_index.size
    return mean

def calculate_accuracy(dftrain,dftest):
    MSE = 0
    for t in range(len(dftest)):
        distances_vector = Euclidean_distance(dftrain,dftest.iloc[[t]])
        predicted_value = knn(dftrain,distances_vector, 5)
        MSE = MSE + pow(predicted_value - float(dftest.iloc[t, 5]),2)
    MSE = MSE/len(dftest)
    return MSE

if __name__ == '__main__':
    data = pd.read_excel('regression.xlsx')
    test = data.iloc[0:int(0.3 * len(data)),:]
    train = data.iloc[int(0.3 * len(data)): len(data),:]
    train = train.reset_index(drop=True)
    # print(calculate_accuracy(train,train))
    print(calculate_accuracy(train,test))