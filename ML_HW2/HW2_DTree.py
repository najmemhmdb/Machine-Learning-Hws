###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import pandas as pd
from sklearn import tree

def preprocess(input_data):
    for j in range(len(input_data.columns)):
        sum = 0
        counter = 0
        for i in range(len(input_data)):
            if(input_data.at[i,j] != '?' and input_data.at[i,j] != ''):
                if input_data.at[i,j] is not None:
                    sum = sum + int(input_data.at[i,j])
                    counter = counter + 1
        sum = sum / counter
        for r in range(len(input_data)):
            if(input_data.at[r,j] == '?' or input_data.at[r,j] == '' ):
                input_data.at[r,j] = int(sum)
            if input_data.at[r,j] is None:
                input_data.at[r,j] = int(sum)
    return input_data

if __name__ == '__main__':
    train_data = pd.read_fwf("breast-cancer-wisconsin-train.data")
    test_data = pd.read_fwf("breast-cancer-wisconsin-test.data")
    train_data.columns = ['column']
    test_data.columns = ['column']
    train_data =train_data["column"].str.split(',', expand=True)
    test_data =test_data["column"].str.split(',', expand=True)
    train_data = preprocess(train_data)
    train_data = train_data.apply(pd.to_numeric, errors='coerce')
    test_data = test_data.apply( pd.to_numeric, errors='coerce' )
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data.iloc[:,0:10], train_data.iloc[:,10:11])
    predicted_value = clf.predict(test_data.iloc[:,0:10])
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i in range(predicted_value.size):
        if (predicted_value[i] == 2 and test_data.iloc[i, 10] == 2):
            TN = TN + 1
        if (predicted_value[i] == 2 and test_data.iloc[i, 10] == 4):
            FN = FN + 1
        if (predicted_value[i] == 4 and test_data.iloc[i, 10] == 2):
            FP = FP + 1
        if (predicted_value[i] == 4 and test_data.iloc[i, 10] == 4):
            TP = TP + 1
    print("accuracy is :")
    print((TN + TP) / predicted_value.size)
    print("TP TN FN FP")
    print(TP, TN, FN, FP)