###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import math
import random
import time


def preprocess(input_data):
    for j in range(len(input_data.columns)):
    # for j in range(5,6):
        sum = 0
        counter = 0
        for i in range(len(input_data)):
        # for i in range(320,341):
            if(input_data.at[i,j] != '?' and input_data.at[i,j] != ''):
                sum = sum + int(input_data.at[i,j])
                counter = counter + 1
        sum = sum / counter
        for r in range(len(input_data)):
            if(input_data.at[r,j] == '?' or input_data.at[r,j] == '' ):
                input_data.at[r,j] = int(sum)
    return input_data

def shuffeling(dataFrame):
    for i in range(0,int(len(dataFrame))):
        r1 = random.randint(0,len(dataFrame) - 1)
        r2 = random.randint(0,len(dataFrame) - 1)
        temp1 = dataFrame.iloc[r1].copy()
        temp2 = dataFrame.iloc[r2].copy()
        dataFrame.iloc[r1] = temp2
        dataFrame.iloc[r2] = temp1
    return dataFrame

def k_fold_cross_validation(input_data,k,p):
    h = int(len(input_data)/k)
    r = len(input_data)%k
    if(p == k ):
        test_data = input_data.iloc[(p-1)*h:p*h + r, :]
        test_data = test_data.reset_index(drop=True)
        train_data = input_data.iloc[0:(p-1)*h , :]
    else:
        test_data = input_data.iloc[(p-1)*h:p*h, :]
        test_data = test_data.reset_index(drop=True)
        if (p == 1):
            train_data = input_data.iloc[p * h:len(input_data), :]
            train_data =train_data.reset_index(drop=True)
        else:
            train_data1 = input_data.iloc[0:(p - 1) * h, :]
            train_data2 = input_data.iloc[p*h:len(input_data), :]
            train_data = train_data1.append(train_data2,ignore_index = True)
    return train_data,test_data

def Euclidean_distance(input_data, test_data):
    d = np.zeros([len(input_data)],dtype=float)
    for i in range(len(input_data)):
        r = 0
        for j in range(len(input_data.columns) - 1):
            r = r + pow(int(input_data.at[i,j]) - int(test_data.iloc[:,j]),2)
        d[i] = math.sqrt(r)
    return d


def similarity(input_data,test_data):
    d = np.zeros(len(input_data))
    B_magnitude = 0
    for r in range(len(input_data.columns) - 1):
        B_magnitude = B_magnitude + pow(int(test_data.iloc[:, r]), 2)
    B_magnitude = math.sqrt(B_magnitude)
    for i in range(len(input_data)):
        sigma = 0
        A_magnitude = 0
        for j in range(len(input_data.columns)-1):
            sigma = sigma + int(input_data.at[i,j]) * int(test_data.iloc[:,j])
            A_magnitude = A_magnitude + pow(int(input_data.at[i,j]),2)
        A_magnitude = math.sqrt(A_magnitude)
        d[i] = sigma/(A_magnitude*B_magnitude)
    return d

def manhattan_distance(input_data,test_data):
    d = np.zeros(len(input_data))
    for i in range(len(input_data)):
        r = 0
        for j in range(len(input_data.columns)-1):
            r = r + abs(int(input_data.at[i,j]) - int(test_data.iloc[:,j]))
        d[i] = r
    return d


def knn(input_data,distances,n, min):
    label = 0
    if(min == 1):
        idx = np.argpartition(distances, n)
        k_min_index = idx[:n]
    elif(min == 0):
        k_min_index = np.argpartition(distances, -1*n)[-1*n:]
    positive = 0
    negative = 0
    for i in k_min_index:
        if(int(input_data.at[i,5]) == 0):
            negative = negative + 1
        elif(int(input_data.at[i,5]) == 1):
            positive = positive + 1
    if (positive > negative):
        label = 1
    return label

def calculate_accuracy(dftrain,dftest):
    TPs = [0,0,0,0,0,0]
    TNs = [0,0,0,0,0,0]
    FPs = [0,0,0,0,0,0]
    FNs = [0,0,0,0,0,0]
    errors = [0,0,0,0,0,0]
    predicted_values = [0,0,0,0,0,0]
    for t in range(len(dftest)):
        r = 0;
        # distances_vector = Euclidean_distance(dftrain,dftest.iloc[[t]])   # with this you must set min = 1 in call knn function
        # distances_vector = manhattan_distance(dftrain,dftest.iloc[[t]])     # with this you must set min = 1 in call knn function
        distances_vector = similarity(dftrain,dftest.iloc[[t]])           # with this you must set min = 0 in call knn function
        for i in [1,3,5,7,15,30]:
            predicted_values[r] = knn(dftrain,distances_vector, i, 0)
            r=r+1
        for o in range(6):
            if (predicted_values[o] - int(dftest.iloc[t, 5]) != 0):
                errors[o] = errors[o] + 1
            if(predicted_values[o]==1 and int(dftest.iloc[t,5])==1):
                TPs[o] = TPs[o] + 1
            elif(predicted_values[o]==1 and int(dftest.iloc[t,5])==0):
                FPs[o] = FPs[o] + 1
            elif(predicted_values[o]==0 and int(dftest.iloc[t,5])==0):
                TNs[o] = TNs[o] + 1
            elif(predicted_values[o]==0 and int(dftest.iloc[t,5])==1):
                FNs[o] = FNs[o] + 1
    accuracy = [0,0,0,0,0,0]
    for v in range(6):
        accuracy[v] = 1- (errors[v]/len(dftest))
    return accuracy,FPs,FNs,TNs,TPs

def main(input_data):
    k = 10
    accuracy_array = [0,0,0,0,0,0]
    TP=[0,0,0,0,0,0]
    TN=[0,0,0,0,0,0]
    FP=[0,0,0,0,0,0]
    FN=[0,0,0,0,0,0]
    start_time = time.time()
    for i in range(k):
        train, test = k_fold_cross_validation(input_data, k, i + 1)
        acc,fp,fn,tn,tp = calculate_accuracy(train, test)
        g = 0
        for (item1,item2) in zip(accuracy_array, acc):
            accuracy_array[g] = (item1 + item2)
            g = g + 1
        g1 = 0
        for (tt, t) in zip(TP, tp):
            TP[g1] = (tt + t)
            g1 = g1 + 1
        g2 = 0
        for (ll,l) in zip(TN,tn):
            TN[g2] = (ll + l)
            g2 = g2 + 1
        g3 = 0
        for (zz, z) in zip(FN, fn):
            FN[g3] = (zz + z)
            g3 = g3 + 1
        g4 = 0
        for (mm, m) in zip(FP, fp):
            FP[g4] = (mm + m)
            g4 = g4 + 1
    print("--- %s seconds ---" % (time.time() - start_time))
    print("accuracy k = 1 ")
    print(accuracy_array[0]/10)
    print("TN = %i | TP = %i | FN = %i | FP = %i",TN[0]/10,TP[0]/10,FN[0]/10,FP[0]/10)
    print("accuracy k = 3 ")
    print(accuracy_array[1] / 10)
    print("TN = %i | TP = %i | FN = %i | FP = %i", TN[1] / 10, TP[1] / 10, FN[1] / 10, FP[1] / 10)
    print("accuracy k = 5 ")
    print(accuracy_array[2] / 10)
    print("TN = %i | TP = %i | FN = %i | FP = %i", TN[2] / 10, TP[2] / 10, FN[2] / 10, FP[2] / 10)
    print("accuracy k = 7 ")
    print(accuracy_array[3] / 10)
    print("TN = %i | TP = %i | FN = %i | FP = %i", TN[3] / 10, TP[3] / 10, FN[3] / 10, FP[3] / 10)
    print("accuracy k = 15 ")
    print(accuracy_array[4] / 10)
    print("TN = %i | TP = %i | FN = %i | FP = %i", TN[4] / 10, TP[4] / 10, FN[4] / 10, FP[4] / 10)
    print("accuracy k = 30 ")
    print(accuracy_array[5] / 10)
    print("TN = %i | TP = %i | FN = %i | FP = %i", TN[5] / 10, TP[5] / 10, FN[5] / 10, FP[5] / 10)
    print([x1 / 10 for x1 in TN])
    print("TP")
    print([x2 / 10 for x2 in TP])
    print("FN")
    print([x3 / 10 for x3 in FN])
    print("FP")
    print([x4 / 10 for x4 in FP])
    print("##########################################################")

def skmain(input_data):
    start_time = time.time()
    input_data = input_data.apply( pd.to_numeric, errors='coerce' )
    test_data = input_data.iloc[9 * 96:10 * 96 + 1, :]
    test_data = test_data.reset_index(drop=True)
    train_data = input_data.iloc[0:9 * 96, :]
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_data.iloc[:,0:5].values, train_data.iloc[:,5:6].values)
    predicted_values = neigh.predict(test_data.iloc[:,0:5])
    err = 0
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(len(test_data)):
        if(predicted_values[i] != test_data.at[i,5]):
            err = err + 1
        if (predicted_values[i] == 1 and test_data.at[i, 5]==1):
            tp = tp + 1
        if (predicted_values[i] == 1 and test_data.at[i, 5]==0):
            fp = fp + 1
        if (predicted_values[i] == 0 and test_data.at[i, 5]==0):
            tn = tn + 1
        if (predicted_values[i] == 0 and test_data.at[i, 5]==1):
            fn = fn + 1
    err = err / 97
    print("--- %s seconds ---" % (time.time() - start_time))
    print("fp  fn  tp  tn")
    print(fp,fn,tp,tn)
    return

if __name__ == '__main__':
    data = pd.read_fwf("mammographic_masses.data")
    data.columns = ['column']
    data = data["column"].str.split(',',expand=True)
    data1 = preprocess(data)
    data1 = shuffeling(data1)
    main(data1)
# part C
    # accuracy = 1 - skmain(data1)
    # print(accuracy)