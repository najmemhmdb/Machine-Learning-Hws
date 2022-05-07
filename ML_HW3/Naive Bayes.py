###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################

import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix



buying = pd.DataFrame(index=['vhigh', 'high', 'med','low'],columns=['acc', 'unacc' , 'good','vgood'])
maint =  pd.DataFrame(index=['vhigh', 'high', 'med','low'],columns=['acc', 'unacc' , 'good','vgood'])
doors =  pd.DataFrame(index=['2', '3', '4','5more'],columns=['acc', 'unacc' , 'good','vgood'])
persons =  pd.DataFrame(index=['2', '4', 'more'],columns=['acc', 'unacc' , 'good','vgood'])
lug_boot = pd.DataFrame(index=['small', 'med', 'big'],columns=['acc', 'unacc' , 'good','vgood'])
safety = pd.DataFrame(index=['low', 'med', 'high'],columns=['acc', 'unacc' , 'good','vgood'])
acc = 0
unacc = 0
good = 0
vgood = 0
confusion_matrix = pd.DataFrame(index=['acc', 'unacc', 'good','vgood'],
                                columns=['acc', 'unacc' , 'good','vgood'])


def shuffeling(dataFrame):
    for i in range(0,int(len(dataFrame))):
        r1 = random.randint(0,len(dataFrame) - 1)
        r2 = random.randint(0,len(dataFrame) - 1)
        temp1 = dataFrame.iloc[r1].copy()
        temp2 = dataFrame.iloc[r2].copy()
        dataFrame.iloc[r1] = temp2
        dataFrame.iloc[r2] = temp1
    return dataFrame


def train_phase(dataset):
    global acc,unacc,good,vgood
    acc = len(dataset.query('`class` == "acc"'))
    unacc = len(dataset.query('`class` == "unacc"'))
    good = len(dataset.query('`class` == "good"'))
    vgood = len(dataset.query('`class` == "vgood"'))
    buying.xs('vhigh')['acc'] = len(dataset.query('`buying` == "vhigh" and `class` == "acc"'))
    buying.xs('vhigh')['unacc'] = len(dataset.query('`buying` == "vhigh" and `class` == "unacc"'))
    buying.xs('vhigh')['good'] = len(dataset.query('`buying` == "vhigh" and `class` == "good"'))
    buying.xs('vhigh')['vgood'] = len(dataset.query('`buying` == "vhigh" and `class` == "vgood"'))
    buying.xs('high')['acc'] = len(dataset.query('`buying` == "high" and `class` == "acc"'))
    buying.xs('high')['unacc'] = len(dataset.query('`buying` == "high" and `class` == "unacc"'))
    buying.xs('high')['good'] = len(dataset.query('`buying` == "high" and `class` == "good"'))
    buying.xs('high')['vgood'] = len(dataset.query('`buying` == "high" and `class` == "vgood"'))
    buying.xs('med')['acc'] = len(dataset.query('`buying` == "med" and `class` == "acc"'))
    buying.xs('med')['unacc'] = len(dataset.query('`buying` == "med" and `class` == "unacc"'))
    buying.xs('med')['good'] = len(dataset.query('`buying` == "med" and `class` == "good"'))
    buying.xs('med')['vgood'] = len(dataset.query('`buying` == "med" and `class` == "vgood"'))
    buying.xs('low')['acc'] = len(dataset.query('`buying` == "low" and `class` == "acc"'))
    buying.xs('low')['unacc'] = len(dataset.query('`buying` == "low" and `class` == "unacc"'))
    buying.xs('low')['good'] = len(dataset.query('`buying` == "low" and `class` == "good"'))
    buying.xs('low')['vgood'] = len(dataset.query('`buying` == "low" and `class` == "vgood"'))

    maint.xs('vhigh')['acc'] = len(dataset.query('`maint` == "vhigh" and `class` == "acc"'))
    maint.xs('vhigh')['unacc'] = len(dataset.query('`maint` == "vhigh" and `class` == "unacc"'))
    maint.xs('vhigh')['good'] = len(dataset.query('`maint` == "vhigh" and `class` == "good"'))
    maint.xs('vhigh')['vgood'] = len(dataset.query('`maint` == "vhigh" and `class` == "vgood"'))
    maint.xs('high')['acc'] = len(dataset.query('`maint` == "high" and `class` == "acc"'))
    maint.xs('high')['unacc'] = len(dataset.query('`maint` == "high" and `class` == "unacc"'))
    maint.xs('high')['good'] = len(dataset.query('`maint` == "high" and `class` == "good"'))
    maint.xs('high')['vgood'] = len(dataset.query('`maint` == "high" and `class` == "vgood"'))
    maint.xs('med')['acc'] = len(dataset.query('`maint` == "med" and `class` == "acc"'))
    maint.xs('med')['unacc'] = len(dataset.query('`maint` == "med" and `class` == "unacc"'))
    maint.xs('med')['good'] = len(dataset.query('`maint` == "med" and `class` == "good"'))
    maint.xs('med')['vgood'] = len(dataset.query('`maint` == "med" and `class` == "vgood"'))
    maint.xs('low')['acc'] = len(dataset.query('`maint` == "low" and `class` == "acc"'))
    maint.xs('low')['unacc'] = len(dataset.query('`maint` == "low" and `class` == "unacc"'))
    maint.xs('low')['good'] = len(dataset.query('`maint` == "low" and `class` == "good"'))
    maint.xs('low')['vgood'] = len(dataset.query('`maint` == "low" and `class` == "vgood"'))

    doors.xs('2')['acc'] = len(dataset.query('`doors` == "2" and `class` == "acc"'))
    doors.xs('2')['unacc'] = len(dataset.query('`doors` == "2" and `class` == "unacc"'))
    doors.xs('2')['good'] = len(dataset.query('`doors` == "2" and `class` == "good"'))
    doors.xs('2')['vgood'] = len(dataset.query('`doors` == "2" and `class` == "vgood"'))
    doors.xs('3')['acc'] = len(dataset.query('`doors` == "3" and `class` == "acc"'))
    doors.xs('3')['unacc'] = len(dataset.query('`doors` == "3" and `class` == "unacc"'))
    doors.xs('3')['good'] = len(dataset.query('`doors` == "3" and `class` == "good"'))
    doors.xs('3')['vgood'] = len(dataset.query('`doors` == "3" and `class` == "vgood"'))
    doors.xs('4')['acc'] = len(dataset.query('`doors` == "4" and `class` == "acc"'))
    doors.xs('4')['unacc'] = len(dataset.query('`doors` == "4" and `class` == "unacc"'))
    doors.xs('4')['good'] = len(dataset.query('`doors` == "4" and `class` == "good"'))
    doors.xs('4')['vgood'] = len(dataset.query('`doors` == "4" and `class` == "vgood"'))
    doors.xs('5more')['acc'] = len(dataset.query('`doors` == "5more" and `class` == "acc"'))
    doors.xs('5more')['unacc'] = len(dataset.query('`doors` == "5more" and `class` == "unacc"'))
    doors.xs('5more')['good'] = len(dataset.query('`doors` == "5more" and `class` == "good"'))
    doors.xs('5more')['vgood'] = len(dataset.query('`doors` == "5more" and `class` == "vgood"'))

    persons.xs('2')['acc'] = len(dataset.query('`persons` == "2" and `class` == "acc"'))
    persons.xs('2')['unacc'] = len(dataset.query('`persons` == "2" and `class` == "unacc"'))
    persons.xs('2')['good'] = len(dataset.query('`persons` == "2" and `class` == "good"'))
    persons.xs('2')['vgood'] = len(dataset.query('`persons` == "2" and `class` == "vgood"'))
    persons.xs('4')['acc'] = len(dataset.query('`persons` == "4" and `class` == "acc"'))
    persons.xs('4')['unacc'] = len(dataset.query('`persons` == "4" and `class` == "unacc"'))
    persons.xs('4')['good'] = len(dataset.query('`persons` == "4" and `class` == "good"'))
    persons.xs('4')['vgood'] = len(dataset.query('`persons` == "4" and `class` == "vgood"'))
    persons.xs('more')['acc'] = len(dataset.query('`persons` == "more" and `class` == "acc"'))
    persons.xs('more')['unacc'] = len(dataset.query('`persons` == "more" and `class` == "unacc"'))
    persons.xs('more')['good'] = len(dataset.query('`persons` == "more" and `class` == "good"'))
    persons.xs('more')['vgood'] = len(dataset.query('`persons` == "more" and `class` == "vgood"'))

    lug_boot.xs('small')['acc'] = len(dataset.query('`lug_boot` == "small" and `class` == "acc"'))
    lug_boot.xs('small')['unacc'] = len(dataset.query('`lug_boot` == "small" and `class` == "unacc"'))
    lug_boot.xs('small')['good'] = len(dataset.query('`lug_boot` == "small" and `class` == "good"'))
    lug_boot.xs('small')['vgood'] = len(dataset.query('`lug_boot` == "small" and `class` == "vgood"'))
    lug_boot.xs('med')['acc'] = len(dataset.query('`lug_boot` == "med" and `class` == "acc"'))
    lug_boot.xs('med')['unacc'] = len(dataset.query('`lug_boot` == "med" and `class` == "unacc"'))
    lug_boot.xs('med')['good'] = len(dataset.query('`lug_boot` == "med" and `class` == "good"'))
    lug_boot.xs('med')['vgood'] = len(dataset.query('`lug_boot` == "med" and `class` == "vgood"'))
    lug_boot.xs('big')['acc'] = len(dataset.query('`lug_boot` == "big" and `class` == "acc"'))
    lug_boot.xs('big')['unacc'] = len(dataset.query('`lug_boot` == "big" and `class` == "unacc"'))
    lug_boot.xs('big')['good'] = len(dataset.query('`lug_boot` == "big" and `class` == "good"'))
    lug_boot.xs('big')['vgood'] = len(dataset.query('`lug_boot` == "big" and `class` == "vgood"'))

    safety.xs('low')['acc'] = len(dataset.query('`safety` == "low" and `class` == "acc"'))
    safety.xs('low')['unacc'] = len(dataset.query('`safety` == "low" and `class` == "unacc"'))
    safety.xs('low')['good'] = len(dataset.query('`safety` == "low" and `class` == "good"'))
    safety.xs('low')['vgood'] = len(dataset.query('`safety` == "low" and `class` == "vgood"'))
    safety.xs('med')['acc'] = len(dataset.query('`safety` == "med" and `class` == "acc"'))
    safety.xs('med')['unacc'] = len(dataset.query('`safety` == "med" and `class` == "unacc"'))
    safety.xs('med')['good'] = len(dataset.query('`safety` == "med" and `class` == "good"'))
    safety.xs('med')['vgood'] = len(dataset.query('`safety` == "med" and `class` == "vgood"'))
    safety.xs('high')['acc'] = len(dataset.query('`safety` == "high" and `class` == "acc"'))
    safety.xs('high')['unacc'] = len(dataset.query('`safety` == "high" and `class` == "unacc"'))
    safety.xs('high')['good'] = len(dataset.query('`safety` == "high" and `class` == "good"'))
    safety.xs('high')['vgood'] = len(dataset.query('`safety` == "high" and `class` == "vgood"'))
    return


def test_phase(input,smoothing):
    labels = []
    acc_all = []
    unacc_all = []
    good_all = []
    vgood_all = []
    for row in range(0,len(input)):
        probability = [1,1,1,1]
        b = input.iloc[row]['buying']
        m = input.iloc[row]['maint']
        d = input.iloc[row]['doors']
        p = input.iloc[row]['persons']
        l = input.iloc[row]['lug_boot']
        s = input.iloc[row]['safety']
        if smoothing == 0 :
            probability[0] = buying._get_value(b,'acc') * maint._get_value(m,'acc') * doors._get_value(d,'acc')\
                            * persons._get_value(p,'acc') * lug_boot._get_value(l,'acc')\
                             * safety._get_value(s,'acc')/(pow(acc,5) * 1208)
            probability[1] = buying._get_value(b, 'unacc') * maint._get_value(m, 'unacc') * doors._get_value(d, 'unacc') \
                             * persons._get_value(p, 'unacc') * lug_boot._get_value(l, 'unacc') \
                             * safety._get_value(s, 'unacc') / (pow(unacc, 5) * 1208)
            probability[2] = buying._get_value(b, 'good') * maint._get_value(m, 'good') * doors._get_value(d, 'good') \
                             * persons._get_value(p, 'good') * lug_boot._get_value(l, 'good') \
                             * safety._get_value(s, 'good') / (pow(good, 5) * 1208)
            probability[3] = buying._get_value(b, 'vgood') * maint._get_value(m, 'vgood') * doors._get_value(d, 'vgood') \
                             * persons._get_value(p, 'vgood') * lug_boot._get_value(l, 'vgood') \
                             * safety._get_value(s, 'vgood') / (pow(vgood, 5) * 1208)
        elif smoothing == 1:
            probability[0] = math.log((buying._get_value(b, 'acc') +1)* (maint._get_value(m, 'acc') + 1) * (doors._get_value(d, 'acc') + 1) \
                             * (persons._get_value(p, 'acc')+1) * (lug_boot._get_value(l, 'acc')+1) \
                             * (safety._get_value(s, 'acc')+1) * acc / (pow(acc + 3, 3) * 1208 * pow(acc + 4, 3)))
            probability[1] = math.log((buying._get_value(b, 'unacc') + 1) * (maint._get_value(m, 'unacc') + 1)\
                             * (doors._get_value(d, 'unacc')+1) * (persons._get_value(p, 'unacc')+1)\
                             * (lug_boot._get_value(l, 'unacc') + 1) * (safety._get_value(s, 'unacc') +1)\
                             * unacc / (pow(unacc + 3, 3) * 1208 * pow(unacc + 4, 3)))
            probability[2] = math.log((buying._get_value(b, 'good')+1) * (maint._get_value(m, 'good')+1) *(doors._get_value(d, 'good')+1) \
                             * (persons._get_value(p, 'good')+1) * (lug_boot._get_value(l, 'good')+1) \
                             * (safety._get_value(s, 'good')+1) * good / (pow(good + 3, 3) * 1208 * pow(good + 4, 3)))
            probability[3] = math.log((buying._get_value(b, 'vgood')+1) * (maint._get_value(m, 'vgood')+1) * (doors._get_value(d, 'vgood')+1) \
                             * (persons._get_value(p, 'vgood')+1) * (lug_boot._get_value(l, 'vgood')+1) \
                             * (safety._get_value(s, 'vgood')+1) * vgood/ (pow(vgood + 3, 3) * 1208 * pow(vgood + 4, 3)))
        if max(probability) == probability[0]:
            labels.append('acc')
            ac.append(probability[0])
        elif max(probability) == probability[1]:
            labels.append('unacc')
            unac.append(probability[1])
        elif max(probability) == probability[2]:
            labels.append('good')
            god.append(probability[2])
        elif max(probability) == probability[3]:
            labels.append('vgood')
            vgo.append(probability[3])
        else:
            labels.append("error")
        acc_vs_all = []
        unacc_vs_all = []
        good_vs_all = []
        vgood_vs_all = []
        for i in np.arange(1.5,10 , 0.2):
            if probability[0]*10000>= i :
                acc_vs_all.append("acc")
            else:
                acc_vs_all.append("not acc")
            if probability[1]*10000>= i :
                unacc_vs_all.append("unacc")
            else:
                unacc_vs_all.append("not unacc")
            if probability[2]*10000>= i :
                good_vs_all.append("good")
            else:
                good_vs_all.append("not good")
            if probability[3]*10000>= i :
                vgood_vs_all.append("vgood")
            else:
                vgood_vs_all.append("not vgood")
        acc_all.append(acc_vs_all)
        unacc_all.append(unacc_vs_all)
        good_all.append(good_vs_all)
        vgood_all.append(vgood_vs_all)
    acc_all = np.array(acc_all).T
    unacc_all = np.array(unacc_all).T
    good_all = np.array(good_all).T
    vgood_all = np.array(vgood_all).T
    return labels,acc_all,unacc_all,good_all,vgood_all


def calculate_measures(real,label):
    confusion_matrix.fillna(0, inplace=True)
    for i in range(len(real)):
        confusion_matrix.loc[real.iloc[i]['class'], label[i]] += 1
    # confusion matrix for acc class vs other classes
    TP_acc = confusion_matrix._get_value('acc','acc')
    TN_acc = confusion_matrix._get_value('unacc','unacc') + confusion_matrix._get_value('good','good')\
             + confusion_matrix._get_value('vgood', 'vgood') + confusion_matrix._get_value('unacc','good')\
             + confusion_matrix._get_value('unacc','vgood') + confusion_matrix._get_value('good','vgood')\
             + confusion_matrix._get_value('good','unacc') + confusion_matrix._get_value('vgood','unacc')\
             + confusion_matrix._get_value('vgood','good')
    FP_acc = confusion_matrix._get_value('unacc','acc') + confusion_matrix._get_value('good','acc')\
             + confusion_matrix._get_value('vgood', 'acc')
    FN_acc = confusion_matrix._get_value('acc','unacc') + confusion_matrix._get_value('acc','good')\
             + confusion_matrix._get_value('acc', 'vgood')
    # confusion matrix for unacc class vs other classes
    TP_unacc = confusion_matrix._get_value('unacc', 'unacc')
    TN_unacc = confusion_matrix._get_value('acc', 'acc') + confusion_matrix._get_value('good', 'good') \
             + confusion_matrix._get_value('vgood', 'vgood') + confusion_matrix._get_value('acc', 'good') \
             + confusion_matrix._get_value('acc', 'vgood') + confusion_matrix._get_value('good', 'vgood') \
             + confusion_matrix._get_value('good', 'acc') + confusion_matrix._get_value('vgood', 'acc') \
             + confusion_matrix._get_value('vgood', 'good')
    FP_unacc = confusion_matrix._get_value('acc', 'unacc') + confusion_matrix._get_value('good', 'unacc') \
             + confusion_matrix._get_value('vgood', 'unacc')
    FN_unacc = confusion_matrix._get_value('unacc', 'acc') + confusion_matrix._get_value('unacc', 'good') \
             + confusion_matrix._get_value('unacc', 'vgood')
    # confusion matrix for good class vs other classes
    TP_good = confusion_matrix._get_value('good', 'good')
    TN_good = confusion_matrix._get_value('unacc', 'unacc') + confusion_matrix._get_value('acc', 'acc') \
             + confusion_matrix._get_value('vgood', 'vgood') + confusion_matrix._get_value('unacc', 'acc') \
             + confusion_matrix._get_value('unacc', 'vgood') + confusion_matrix._get_value('acc', 'vgood') \
             + confusion_matrix._get_value('acc', 'unacc') + confusion_matrix._get_value('vgood', 'unacc') \
             + confusion_matrix._get_value('vgood', 'acc')
    FP_good = confusion_matrix._get_value('acc', 'good') + confusion_matrix._get_value('unacc', 'good') \
             + confusion_matrix._get_value('vgood', 'good')
    FN_good = confusion_matrix._get_value('good', 'unacc') + confusion_matrix._get_value('good', 'acc') \
             + confusion_matrix._get_value('good', 'vgood')
    # confusion matrix for vgood class vs other classes
    TP_vgood = confusion_matrix._get_value('vgood', 'vgood')
    TN_vgood = confusion_matrix._get_value('unacc', 'unacc') + confusion_matrix._get_value('good', 'good') \
             + confusion_matrix._get_value('acc', 'acc') + confusion_matrix._get_value('unacc', 'good') \
             + confusion_matrix._get_value('unacc', 'acc') + confusion_matrix._get_value('good', 'acc') \
             + confusion_matrix._get_value('good', 'unacc') + confusion_matrix._get_value('acc', 'unacc') \
             + confusion_matrix._get_value('acc', 'good')
    FP_vgood = confusion_matrix._get_value('unacc', 'vgood') + confusion_matrix._get_value('good', 'vgood') \
             + confusion_matrix._get_value('acc', 'vgood')
    FN_vgood = confusion_matrix._get_value('vgood', 'unacc') + confusion_matrix._get_value('vgood', 'good') \
             + confusion_matrix._get_value('vgood', 'acc')
    TPR = (TP_acc + TP_good + TP_unacc + TP_vgood) / (TP_acc + TP_good + TP_unacc + TP_vgood + FN_acc + FN_good + FN_unacc + FN_vgood)
    print("sensivity : ")
    print(TPR)
    specificity = (TN_acc + TN_good + TN_unacc + TN_vgood) / (TN_acc + TN_good + TN_unacc + TN_vgood + FP_acc + FP_good + FP_unacc + FP_vgood)
    print("specificity :")
    print(specificity)
    FPR = (FP_acc + FP_good + FP_unacc + FP_vgood) / (TN_acc + TN_good + TN_unacc + TN_vgood + FP_acc + FP_good + FP_unacc + FP_vgood)
    print("False Positive Rate : ")
    print(FPR)
    FNR = (FN_acc + FN_good + FN_unacc + FN_vgood) / (TP_acc + TP_good + TP_unacc + TP_vgood + FN_acc + FN_good + FN_unacc + FN_vgood)
    print("False Negative Rate : ")
    print(FNR)

    print("confusion matrix : ")
    print(confusion_matrix)
    return TPR,FPR

def calculate_measures_2class(predict, true,label):
    fp = 0
    tp = 0
    p = 0
    n = 0
    for i in range(519):
        if true.iloc[i]['class']==label:
            p+=1
            if predict[i] == label:
                tp+=1
        elif true.iloc[i]['class']!=label:
            n+=1
            if predict[i] == label:
                fp+=1
    return tp/p,fp/n


def ROC(acc_vs_all,unacc_vs_all,good_vs_all,vgood_vs_all,test):
    x1 = [1]
    x2 = [1]
    x3=[1]
    x4 = [1]
    y1= [1]
    y2= [1]
    y3= [1]
    y4 = [1]
    for i in range(acc_vs_all.shape[0]):
        tpr1,fpr1 = calculate_measures_2class(acc_vs_all[i], test,"acc")
        tpr2,fpr2 = calculate_measures_2class(unacc_vs_all[i], test,"unacc")
        tpr3,fpr3 = calculate_measures_2class(good_vs_all[i], test,"good")
        tpr4,fpr4 = calculate_measures_2class(vgood_vs_all[i], test,"vgood")
        x1.append(fpr1)
        y1.append(tpr1)
        x2.append(fpr2)
        y2.append(tpr2)
        x3.append(fpr3)
        y3.append(tpr3)
        x4.append(fpr4)
        y4.append(tpr4)
    plt.plot(x1,y1,"red")
    plt.plot(x2,y2,"blue")
    plt.plot(x3,y3,"green")
    plt.plot(x4,y4,"orange")
    plt.title('ROC', fontsize=24)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()

if __name__ == '__main__':
    data = pd.read_fwf("car.data")
    data.columns = ['column']
    data = data["column"].str.split(',', expand=True)
    data = data.rename(columns={0: 'buying', 1: 'maint', 2:'doors', 3:'persons',4:'lug_boot',5:'safety',6:'class'})
    data.xs(100)['class'] = "unacc"
    data = shuffeling(data)
    split_point = int(len(data)*0.7)
    train = data.iloc[:split_point, :]
    test = data.iloc[split_point:, :]
    train_phase(train)
    predict,acc_vs_all,unacc_vs_all,good_vs_all,vgood_vs_all = test_phase(test, smoothing=0)
    calculate_measures(test,predict)
    ROC(acc_vs_all,unacc_vs_all,good_vs_all,vgood_vs_all,test)