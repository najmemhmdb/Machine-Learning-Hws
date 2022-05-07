###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


def shuffeling(dataFrame):
    for i in range(0,int(dataFrame.size/2)):
        r1 = random.randint(0,dataFrame.size/2 - 1)
        r2 = random.randint(0,dataFrame.size/2 - 1)
        x = dataFrame.at[r1,'x']
        y = dataFrame.at[r1,'y']
        dataFrame.at[r1,'x'] = dataFrame.at[r2,'x']
        dataFrame.at[r1,'y'] = dataFrame.at[r2,'y']
        dataFrame.at[r2,'x'] = x
        dataFrame.at[r2,'y'] = y
    return dataFrame

def normalizer(dataFrame):
    maxX = dataFrame['x'].max()
    minX = dataFrame['x'].min()
    maxY = dataFrame['y'].max()
    minY = dataFrame['y'].min()
    dataFrame['x'] = pd.to_numeric(dataFrame['x'], downcast="float")
    dataFrame['y'] = pd.to_numeric(dataFrame['y'], downcast="float")
    for i in range(0,int(dataFrame.size / 2)):
        dataFrame.at[i,'x'] = (dataFrame.at[i,'x'] - minX )/(maxX - minX)
        dataFrame.at[i,'y'] = (dataFrame.at[i,'y'] - minY )/(maxY - minY)
    return dataFrame

def regression(dataFrame,d,n,alpha,lambda_1):
    # initiate theta vector
    theta = np.ones((d+1,), dtype=float)

    # number of data
    m = int(dataFrame.size / 2)
    for t in range(0,n):
        g = calculate_Gradiant_vector(theta,dataFrame,d,lambda_1)
        theta[:] = theta[:] - alpha * g[:]
    return theta

def calculate_Gradiant_vector(theta,dataFrame,d, lambda_1):
    # print(dataFrame)
    gradiantVector = np.zeros((d+1,), dtype=float)
    m = int(dataFrame.size/2)
    for k in range(0,d+1):
        sigma = 0
        # calculate sigma
        for i in range(0,m):
            sigma1 = 0
            for j in range(0,d+1):
                sigma1 = pow(dataFrame.at[i,'x'],j)*theta.__getitem__(j) + sigma1
            sigma1 = sigma1 - dataFrame.at[i,'y']
            sigma = sigma1 * pow(dataFrame.at[i,'x'],k) + sigma
        sigma = (sigma + (lambda_1 * k)) / m
        gradiantVector.__setitem__(k, sigma)
    return gradiantVector

def predict(df,theta,index):
    m = int(df.size / 2)
    predicted_values = np.zeros((m,), dtype=float)
    for i in range(index,index + m):
        value = 0
        for j in range(0,theta.size):
            value = value + theta.__getitem__(j) * pow(df.at[i,'x'],j)
        predicted_values.__setitem__(i - index,value)
    return predicted_values

def calculate_error(predicted,real , index):
     error = 0
     for i in range(0,predicted.size):
         error = pow((predicted.__getitem__(i) - real.at[index + i , 'y']) , 2) + error
     error = error / predicted.size
     return error

def normal_equation(X,Y,d):
    X0 = np.ones((int(X.size),), dtype=float)
    X1 = X.to_numpy()
    Y = Y.to_numpy()
    X = np.column_stack((X0,X1))
    Xi = X1
    for i in range(2,d+1):
        Xi = Xi * X1
        X = np.column_stack((X,Xi))
    XT = np.transpose(X)
    XTX = XT.dot(X)
    inverse = np.linalg.inv(XTX)
    invXT = inverse.dot(XT)
    theta = invXT.dot(Y)
    return theta
def main_part1():
# read dataset
    df = pd.read_csv('Dataset1.csv', delimiter=',', encoding="utf-8-sig")

# shuffling dataset
    df = shuffeling(df)

# normalizing dataset
    df = normalizer(df)

# split train and test data
    m = int(df.size / 2)
    s = int(0.7 * m)
    train = df.iloc[:s, :]
    test = df.iloc[s:, :]

# call regression function
    fixed_theta1 = regression(train, 1, 15000, 0.1, 0.5)
    print("first")
    fixed_theta2 = regression(train, 2, 15000, 0.1, 0.5)
    print("second")
    fixed_theta3 = regression(train, 3, 15000, 0.1, 0.5)
    print("third")
    fixed_theta4 = regression(train, 4, 15000, 0.1, 0.5)
    print("fourth")
    fixed_theta5 = regression(train, 5, 15000, 0.1, 0.5)
    print("fifth")
#
# call normal equation function
#     fixed_theta1 = normal_equation(df['x'],df['y'],1)
#     fixed_theta2 = normal_equation(df['x'],df['y'],4)
#     fixed_theta3 = normal_equation(df['x'],df['y'],6)
#     fixed_theta4 = normal_equation(df['x'],df['y'],8)
#     fixed_theta5 = normal_equation(df['x'],df['y'],10)
# # predict the output value of test
    predicted_values1 = predict(test, fixed_theta1,s)
    predicted_values2 = predict(test, fixed_theta2,s)
    predicted_values3 = predict(test, fixed_theta3,s)
    predicted_values4 = predict(test, fixed_theta4,s)
    predicted_values5 = predict(test, fixed_theta5,s)
    values1 = predict(train, fixed_theta1, 0)
    values2 = predict(train, fixed_theta1, 0)
    values3 = predict(train, fixed_theta1, 0)
    values4 = predict(train, fixed_theta1, 0)
    values5 = predict(train, fixed_theta1, 0)

# #calculate MSE error
    MSE1 = calculate_error(predicted_values1, test,s)
    MSE2 = calculate_error(predicted_values2, test,s)
    MSE3 = calculate_error(predicted_values3, test,s)
    MSE4 = calculate_error(predicted_values4, test,s)
    MSE5 = calculate_error(predicted_values5, test,s)
    MSE1T = calculate_error(values1, train, 0)
    MSE2T = calculate_error(values2, train, 0)
    MSE3T = calculate_error(values3, train, 0)
    MSE4T = calculate_error(values4, train, 0)
    MSE5T = calculate_error(values5, train, 0)
    print("test error")
    print("MSE1 = " + str(MSE1))
    print("MSE2 = " + str(MSE2))
    print("MSE3 = " + str(MSE3))
    print("MSE4 = " + str(MSE4))
    print("MSE5 = " + str(MSE5))
    print("train error")
    print("MSE1 = " + str(MSE1T))
    print("MSE2 = " + str(MSE2T))
    print("MSE3 = " + str(MSE3T))
    print("MSE4 = " + str(MSE4T))
    print("MSE5 = " + str(MSE5T))
    print("theta vector")
    print(fixed_theta1)
    print(fixed_theta2)
    print(fixed_theta3)
    print(fixed_theta4)
    print(fixed_theta5)
# display outputs
    fig, axs = plt.subplots(2, 3, constrained_layout=True)
    axs[0][0].plot(test['x'], test['y'], "." , color = "red")
    axs[0][1].plot(test['x'], test['y'], "o" , color = "red")
    axs[0][2].plot(test['x'], test['y'], "o" , color = "red")
    axs[1][0].plot(test['x'], test['y'], "o" , color = "red")
    axs[1][1].plot(test['x'], test['y'], "o" , color = "red")
    axs[0][0].plot(test['x'], predicted_values1,"x",color="yellow")
    axs[0][1].plot(test['x'], predicted_values2,"x",color="aqua")
    axs[0][2].plot(test['x'], predicted_values3,"x",color="tomato")
    axs[1][0].plot(test['x'], predicted_values4,"x",color="pink")
    axs[1][1].plot(test['x'], predicted_values5,"x",color="lime")
    axs[0][0].set_xlabel('x values')
    axs[0][0].set_ylabel('y values')
    axs[0][1].set_xlabel('x values')
    axs[0][1].set_ylabel('y values')
    axs[0][2].set_xlabel('x values')
    axs[0][2].set_ylabel('y values')
    axs[1][0].set_xlabel('x values')
    axs[1][0].set_ylabel('y values')
    axs[1][1].set_xlabel('x values')
    axs[1][1].set_ylabel('y values')
    fig.suptitle('Display Test data with predicted values', fontsize=16)
    plt.show()
    return

def main_part2():
# read dataset
    df = pd.read_csv('Dataset2.csv', delimiter=',', encoding="utf-8-sig")
# shuffling dataset
    df = shuffeling(df)
# normalizing dataset
    df = normalizer(df)
# split train and test data
    m = int(df.size / 2)
    s = int(0.7 * m)
    train = df.iloc[:s, :]
    test = df.iloc[s:, :]
# call regression function
    fixed_theta = regression(train, 1, 1500, 0.01, 0)
# predict the output value of test
    predicted_values = predict(test, fixed_theta,s)
#calculate MSE error
    MSE = calculate_error(predicted_values, test,s)
    print("MSE = " + str(MSE))
# display outputs
    fig, axs = plt.subplots(1,1, constrained_layout=True)
    axs.plot(test['x'], test['y'], "." , color = "red")
    axs.plot(test['x'], predicted_values,"x",color="pink")
    axs.set_xlabel('x values')
    axs.set_ylabel('y values')
    fig.suptitle('Display Test data with predicted values', fontsize=16)
    plt.show()



    return

if __name__ == "__main__":
    main_part1()
    # main_part2()
