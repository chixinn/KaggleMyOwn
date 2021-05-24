##COPYRIGHT from 仲益教育

import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from Help_functions_v2 import RMSE
import warnings
warnings.filterwarnings("ignore")

#1: import data
train_data=pd.read_csv("C:/Users/Haixiang He/Desktop/仲益/Kaggle/data/train_clean2.csv")

#1.2: split data into training and testing
train_data.index=train_data["Id"]
train_data.drop("Id", axis=1, inplace=True)
dependentV=train_data["SalePrice"]
train_data.drop("SalePrice", axis=1, inplace=True)
#Partition the dataset in train + validation sets
#usually linear regression needs at least 30 observations
#split of train and validation can be 70:30, or 60:40
X_train, X_test, y_train, y_test = train_test_split(train_data, dependentV, test_size = 0.3, random_state = 0)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))

#2.1 fit tree model
treeModel=DecisionTreeRegressor(max_depth=5)
treeModel.fit(X_train,y_train)
y_trainPred=treeModel.predict(X_train)
#in sample
print("decision tree in-sample r-squared is")
print(r2_score(y_train, y_trainPred))
print ("decision tree RMSE is of in-sample")
print(RMSE(y_trainPred,y_train))
#out of sample
y_testPred=treeModel.predict(X_test)
print("decision tree out-of-sample r-squared is")
print(r2_score(y_test, y_testPred))
print ("decision tree RMSE is out-of-sample")
print(RMSE(y_testPred,y_test))

#2.2 fit a adaboost model
adaBoostModel = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=400, learning_rate=0.5)
adaBoostModel.fit(X_train, y_train)
y_trainPred2=adaBoostModel.predict(X_train)
#in sample
print("adaboost in-sample r-squared is")
print(r2_score(y_train, y_trainPred2))
print ("adaboost RMSE is of in-sample")
print(RMSE(y_trainPred2,y_train))
#out of sample
y_testPred2=adaBoostModel.predict(X_test)
print("adaboost out-of-sample r-squared is")
print(r2_score(y_test, y_testPred2))
print ("adaboost RMSE is out-of-sample")
print(RMSE(y_testPred2,y_test))

#2.3 fit a bagging model
baggingModel = BaggingRegressor(DecisionTreeRegressor(max_depth=5),n_estimators=200)
baggingModel.fit(X_train, y_train)
y_trainPred3=baggingModel.predict(X_train)
#in sample
print("bagging in-sample r-squared is")
print(r2_score(y_train, y_trainPred3))
print ("bagging RMSE is of in-sample")
print(RMSE(y_trainPred3,y_train))
#out of sample
y_testPred3=baggingModel.predict(X_test)
print("bagging out-of-sample r-squared is")
print(r2_score(y_test, y_testPred3))
print ("bagging RMSE is out-of-sample")
print(RMSE(y_testPred3,y_test))