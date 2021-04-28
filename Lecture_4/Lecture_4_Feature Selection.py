##COPYRIGHT from 仲益教育
import sys
sys.path.append("C://Users//Haixiang He//desktop//Kaggle//Lecture 2")
#conda install mxltend --channel conda-forge
#pip install package name
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression, SelectKBest
import numpy as np
from Help_functions_v2 import sklearn_Pvalue, sklearn_adjR2, RMSE

#1: import data
train_data=pd.read_csv("C:/Users/Haixiang He/Desktop/Kaggle/data/train_clean2.csv")

#2.1: feature selection with p_value
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

#2.1.1 select top 20 features with the best F-stats
# 固定20个features，找出20个features
# 应用sk-learn中的SelectKBest
X_scored = SelectKBest(score_func=f_regression, k=20)
X_scored.fit(X_train, y_train)
feat_list=X_scored.get_support()

feature_scoring = pd.DataFrame({
        'feature': X_train.columns[feat_list],
        'pvalue': X_scored.pvalues_[feat_list]
    })

print(feature_scoring)
#homework, find out the RMSE of in sample and out of sample regression of the top 20 selected features

#2.1.2 select all features with individual p_value <=0.05
X_scored2 = SelectKBest(score_func=f_regression, k='all').fit(X_train, y_train)

feature_scoring2 = pd.DataFrame({
        'feature': X_train.columns,
        'pvalue': X_scored2.pvalues_
    })
feat_pvalue_significant=feature_scoring[feature_scoring2.pvalue<=0.05]
feat_pvalue_significant['feature'].values

#2.1.3 select N features based on the 
#2.2 feature selection based on forward/backward elimination based on R Square
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
#2.2.1 find the best k features using stepforward method
stepforward = SFS(LinearRegression(), 
           k_features=10, # 加到10个features使r-square更好
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='r2',
           cv=0)

stepforward = stepforward.fit(np.array(X_train), y_train)
print(X_train.columns[list(stepforward.k_feature_idx_)])
#homework, find out the RMSE of in sample and out of sample regression of selected features 
#using forward elimination

#2.2.2 find the best k features using stepbackward method
backward = SFS(LinearRegression(), 
           k_features=10, 
           forward=False, 
           floating=False, 
           verbose=2,
           scoring='r2',
           cv=0)

backward = stepforward.fit(np.array(X_train), y_train)
print(X_train.columns[list(backward.k_feature_idx_)])


#3 feature selection with regularization
#3.1 Ridge Regularization
ridge_model=Ridge(alpha=1)# specify这里的惩罚是多少，下一节课会去讲如何做lambda的选择
ridge_model.fit(X_train, y_train)
y_trainPred=ridge_model.predict(X_train)
#in sample
print("in-sample r-squared is")
print(r2_score(y_train, y_trainPred))
print ("RMSE is of in-sample")
print(RMSE(y_trainPred,y_train))
#out of sample
y_testPred=ridge_model.predict(X_test)
print("out-of-sample r-squared is")
print(r2_score(y_test, y_testPred))
print ("RMSE is out-of-sample")
print(RMSE(y_testPred,y_test))

#3.2 Lasso Regularization
lasso_model=Lasso(alpha=1)
lasso_model.fit(X_train, y_train)
y_trainPred=lasso_model.predict(X_train)
#in sample
print("lasso in-sample r-squared is")
print(r2_score(y_train, y_trainPred))
print ("Lasso RMSE of in-sample is ")
print(RMSE(y_trainPred,y_train))
#out of sample
y_testPred=lasso_model.predict(X_test)
print("Lasso out-of-sample r-squared is")
print(r2_score(y_test, y_testPred))
print ("Lasso RMSE out-of-sample is")
print(RMSE(y_testPred,y_test))
