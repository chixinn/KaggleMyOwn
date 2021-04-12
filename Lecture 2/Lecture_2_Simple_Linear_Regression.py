##COPYRIGHT from 仲益教育
import sys
#to help find the path of the help function
sys.path.append("C://Users//Haixiang He//desktop//Kaggle//Lecture 2")
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score #去计算R方的
from sklearn.model_selection import train_test_split
import numpy as np
from Help_functions_v2 import sklearn_Pvalue, RMSE # help_functions是用来算p-value的
import seaborn as sns
import matplotlib.pyplot as plt

#Before we build model, let's make sure what is our target
#R#calcualte RMSE -- root mean squared error

##Part 2: Simple Linear Regression
#1.1: import data
train_data=pd.read_csv("C:/Users/Haixiang He/Desktop/Kaggle/data/train_clean.csv")
train_data.index=train_data["Id"]
train_data.drop("Id", axis=1, inplace=True)
# dependent variable // independent variable
dependentV=train_data["SalePrice"]
IndependentV=train_data.drop(["SalePrice"],axis=1)
#1.2: dummy variables: categorical => numerical

IndependentV=pd.get_dummies(IndependentV)

#2.1 split in-sample data to training and validation sets
# Partition the dataset in train + validation sets
# usually linear regression needs at least 30 observations
# split of train and validation can be 70:30, or 60:40
X_train, X_test, y_train, y_test = train_test_split(IndependentV, dependentV, test_size = 0.3, random_state = 0)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))

#2.2 fit simple linear regression on the in-sample data
#the most correlated variables in the hit map analysis
corrmat = train_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#top correlated variables
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                 yticklabels=cols.values, xticklabels=cols.values)
#['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# 先建立一个空的linearRegression
lr=LinearRegression()
#linearRegression function requires the reshape the series if your data has a single feature
lr.fit(X_train["GrLivArea"].values.reshape(-1,1), y_train)

#2.3 R^2, p_value, RMSE
y_trainPred=lr.predict(X_train["GrLivArea"].values.reshape(-1,1))
print("in-sample r-squared is")
r2_score(y_train, y_trainPred)
p_value=sklearn_Pvalue(X_train["GrLivArea"].values.reshape(-1,1), y_train)
print ("P-value Table of in-sample")
print(p_value)
print ("RMSE is of in-sample")
print(RMSE(y_trainPred,y_train))

#2.4 look at the prediction on the validation set
y_testPred = lr.predict(X_test["GrLivArea"].values.reshape(-1,1))
print("out-of-sample r-squared is")
r2_score(y_test, y_testPred)
p_value=sklearn_Pvalue(X_test["GrLivArea"].values.reshape(-1,1), y_test)
print ("P-value Table of out-of-sample")
print(p_value)
print ("RMSE is of out-of-sample")
print(RMSE(y_testPred,y_test))

# Plot residuals
plt.scatter(y_trainPred, y_trainPred - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_testPred, y_testPred - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()
# Plot predictions
plt.scatter(y_trainPred, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_testPred, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

#3.1 linear regression on dummy variable
lr2=LinearRegression()
#linearRegression function requires the reshape the series if your data has a single feature
lr2.fit(X_train["CentralAir_Y"].values.reshape(-1,1), y_train)

#3.2 R^2, p_value, RMSE
y_trainPred2=lr2.predict(X_train["CentralAir_Y"].values.reshape(-1,1))
print("in-sample r-squared is")
r2_score(y_train, y_trainPred2)
RMSE(y_trainPred2, y_train)
p_value2=sklearn_Pvalue(X_train["CentralAir_Y"].values.reshape(-1,1), y_train)
print ("P-value Table of in-sample")
print(p_value2)
print ("RMSE is of in-sample")
print(RMSE(y_trainPred2,y_train))


#3.3 look at the prediction on the validation set
y_testPred2 = lr2.predict(X_test["CentralAir_Y"].values.reshape(-1,1))
print("out-of-sample r-squared is")
r2_score(y_test, y_testPred2)
p_value2=sklearn_Pvalue(X_test["CentralAir_Y"].values.reshape(-1,1), y_test)
print ("P-value Table of out-of-sample")
print(p_value2)
print ("RMSE is of out-of-sample")
print(RMSE(y_testPred2,y_test))

# Plot residuals
plt.scatter(y_trainPred2, y_trainPred2 - y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_testPred2, y_testPred2 - y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()
# Plot predictions
plt.scatter(y_trainPred2, y_train, c = "blue", marker = "s", label = "Training data")
plt.scatter(y_testPred2, y_test, c = "lightgreen", marker = "s", label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()


