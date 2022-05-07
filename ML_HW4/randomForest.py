###############################################
# Student_ID : 99131009                       #
# Name : Najmeh                               #
# Last Name : Mohammadbagheri                 #
# E-mail : Najmeh.mohammadbagheri77@gmail.com #
###############################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

if __name__ == '__main__':
    data= pd.read_csv("pima_indians_diabetes.csv")
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,0:8], data.iloc[:,8], test_size=0.2, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    cls = RandomForestClassifier(n_estimators=100,max_depth=4,max_features=8)
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    print('accuracy  :', metrics.accuracy_score(y_test, y_pred))

# part C
# Adaboost
    print("-------------------------------------------------------")
    print("part C")
    ada = AdaBoostClassifier(n_estimators=200)
    ada.fit(X_train,y_train)
    y_pred = ada.predict(X_test)
    print('accuracy  adaboost :', metrics.accuracy_score(y_test, y_pred))



# Bagging
    bag = BaggingClassifier(LogisticRegression(),n_estimators=20,bootstrap=True)
    bag.fit(X_train,y_train)
    y_pred = bag.predict(X_test)
    print('accuracy  Bagging :', metrics.accuracy_score(y_test, y_pred))



# SVM RBF GAMMA = 0.01
    svclassifier = SVC(kernel='rbf',gamma=0.01)
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    print("accuracy SVM RBF: ",  metrics.accuracy_score(y_test, y_pred))