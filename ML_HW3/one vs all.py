###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################

import os
import struct
import numpy as np
from array import array as pyarray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import multilabel_confusion_matrix
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def load_mnist(image_path,label_path,digits=np.arange(10)):
    flbl = open(label_path, 'rb')
    magic_nr, size = struct.unpack(">ii", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(image_path, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">iiii", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
    ind = [k for k in range(size) if lbl[k] in digits]
    N = size  # int(len(ind) * size/100.)
    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(N):  # int(len(ind) * size/100.)):
        images[i] = np.array(img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]) \
            .reshape((rows, cols))
        labels[i] = lbl[ind[i]]
    labels = [label[0] for label in labels]
    return images, labels


def convert_img_to_vector(input):
    output = []
    for i in range(input.shape[0]):
        output.append(input[i].flatten())
    return np.array(output)

def train_phase(images,labels):
    # classifier for #0 vs all
    labels_0 = []
    for i in range(len(labels)):
        if labels[i] != 0:
            labels_0.append(0)
        else:
            labels_0.append(1)
    clf_0 = LogisticRegression(random_state=0).fit(images, labels_0)

    # classifier for #1 vs all
    labels_1 = []
    for i in range(len(labels)):
        if labels[i] != 1 :
            labels_1.append(0)
        else:
            labels_1.append(1)
    clf_1 = LogisticRegression(random_state=0).fit(images, labels_1)
    # classifier for #2 vs all
    labels_2 = []
    for i in range(len(labels)):
        if labels[i] != 2:
            labels_2.append(0)
        else:
            labels_2.append(1)
    clf_2 = LogisticRegression(random_state=0).fit(images, labels_2)
    # classifier for #3 vs all
    labels_3 = []
    for i in range(len(labels)):
        if labels[i] != 3:
            labels_3.append(0)
        else:
            labels_3.append(1)
    clf_3 = LogisticRegression(random_state=0).fit(images, labels_3)
    # classifier for #4 vs all
    labels_4 = []
    for i in range(len(labels)):
        if labels[i] != 4:
            labels_4.append(0)
        else:
            labels_4.append(1)
    clf_4 = LogisticRegression(random_state=0).fit(images, labels_4)
    # classifier for #5 vs all
    labels_5 = []
    for i in range(len(labels)):
        if labels[i] != 5:
            labels_5.append(0)
        else:
            labels_5.append(1)
    clf_5 = LogisticRegression(random_state=0).fit(images, labels_5)
    # classifier for #6 vs all
    labels_6 = []
    for i in range(len(labels)):
        if labels[i] != 6:
            labels_6.append(0)
        else:
            labels_6.append(1)
    clf_6 = LogisticRegression(random_state=0).fit(images, labels_6)
    # classifier for #7 vs all
    labels_7 = []
    for i in range(len(labels)):
        if labels[i] != 7:
            labels_7.append(0)
        else:
            labels_7.append(1)
    clf_7 = LogisticRegression(random_state=0).fit(images, labels_7)
    # classifier for #8 vs all
    labels_8 = []
    for i in range(len(labels)):
        if labels[i] != 8:
            labels_8.append(0)
        else:
            labels_8.append(1)
    clf_8 = LogisticRegression(random_state=0).fit(images, labels_8)
    # classifier for #9 vs all
    labels_9 = []
    for i in range(len(labels)):
        if labels[i] != 9:
            labels_9.append(0)
        else:
            labels_9.append(1)
    clf_9 = LogisticRegression(random_state=0).fit(images, labels_9)
    all_clcs = [clf_0,clf_1,clf_2,clf_3,clf_4,clf_5,clf_6,clf_7,clf_8,clf_9]
    return all_clcs


def test_phase(clfs,Timgs):
    predict = []
    # print(clfs[7].predict_proba(Timgs[0].reshape(1, -1))[0][1])
    for i in range(Timgs.shape[0]):
        array = [clfs[0].predict_proba(Timgs[i].reshape(1, -1))[0][1],
                 clfs[1].predict_proba(Timgs[i].reshape(1, -1))[0][1],
                 clfs[2].predict_proba(Timgs[i].reshape(1, -1))[0][1],
                 clfs[3].predict_proba(Timgs[i].reshape(1, -1))[0][1],
                 clfs[4].predict_proba(Timgs[i].reshape(1, -1))[0][1],
                 clfs[5].predict_proba(Timgs[i].reshape(1, -1))[0][1],
                 clfs[6].predict_proba(Timgs[i].reshape(1, -1))[0][1],
                 clfs[7].predict_proba(Timgs[i].reshape(1, -1))[0][1],
                 clfs[8].predict_proba(Timgs[i].reshape(1, -1))[0][1],
                 clfs[9].predict_proba(Timgs[i].reshape(1, -1))[0][1]]
        max_probability = np.argmax(array)
        predict.append(max_probability)
    return np.array(predict)


def confusion_matrix(predict,true):
    print(multilabel_confusion_matrix(true, predict))


if __name__ == '__main__':
    images,labels = load_mnist('train-images.idx3-ubyte','train-labels.idx1-ubyte')
    Timages, Tlabels = load_mnist('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte')
    # img_vector = convert_img_to_vector(images)
    # classifiers = train_phase(img_vector,labels)
    # predicted_values = test_phase(classifiers,convert_img_to_vector(images))
    # np.savetxt('train_labels.txt',predicted_values)
    error = 0
    predicted = np.loadtxt('labels.txt')
    for i in range(predicted.shape[0]):
        if predicted[i] != Tlabels[i]:
            error += 1
    print(error/len(Tlabels))
    confusion_matrix(predicted,Tlabels)
    randomi = np.random.randint(10000,size=(25))
    randomi = randomi.reshape(5,5)
    fig,ax = plt.subplots(5,5)
    for i in range(5):
        for j in range(5):
            # ax[i][j].plot(i,j)
            ax[i][j].imshow(Timages[i*5+j])
            ax[i][j].axes.xaxis.set_visible(False)
            ax[i][j].axes.yaxis.set_visible(False)
            ax[i][j].set_title("pre:" + str(int(predicted[5*i+j])) + " / true:" + str(Tlabels[5*i+j]),size=8)
    plt.show()
