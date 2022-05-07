###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def my_kernel(x,r,d):
    tmp = x
    for i in range(d-1):
        tmp = np.dot(tmp.T,tmp)
        print(tmp.shape)
    return tmp


if __name__ == '__main__':
    data = pd.read_fwf("parkinsons.data")
    print(data)
    data.columns = ['column']
    data = data["column"].str.split(',', expand=True)
    data = shuffle(data)
    temp = data.loc[:, data.columns != 0]
    temp = temp.astype(float)
    X_train, X_test, Y_train, Y_test = train_test_split(temp.loc[:, temp.columns != 17],
                                                        data[17], test_size=0.25, random_state=42)
# ####################################################################
# linear kernel
#     svclassifier = SVC(kernel='linear',coef0=60)
# polynomial Kernel
#     svclassifier = SVC(kernel='poly',degree=4,coef0=1,max_iter = 1e5)
# RBF Kernel
#     svclassifier = SVC(kernel='rbf', gamma=10)
# Sigmoid Kernel
    svclassifier = SVC(kernel='sigmoid',coef0=0.0001)
# ####################################################################
    svclassifier.fit(X_train, Y_train)
    y_pred = svclassifier.predict(X_test)
    i = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    Y_test= Y_test.to_numpy()
    for l in y_pred:
        if l == '0' and Y_test[i] == '0':
            TN += 1
        elif l == '1' and Y_test[i] == '1':
            TP += 1
        elif l == '1' and Y_test[i] == '0':
            FP += 1
        elif l == '0' and Y_test[i] == '1':
            FN += 1
        i +=1
    print(TN,TP,FN,FP)
    print("Accuracy : ")
    print((TP + TN) /(TP + TN + FP + FN))
    print("Fscore :")
    print(2*TP / (2*TP + FN + FP))

    #     # svclassifier = SVC(kernel='precomputed',max_iter = 1e5)
    #     # kernel_train = my_kernel(X_train,Y_train,r = 2,d=2)
    #     # svclassifier.fit(kernel_train, Y_train)
    #     # kernel_test = my_kernel(X_test,Y_test,r =2,d=2)
    #     # y_pred = svclassifier.predict(kernel_test)
