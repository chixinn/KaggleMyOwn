##COPYRIGHT from 仲益教育
import sys
sys.path.append("C://Users//Haixiang He//desktop//Kaggle//Lecture 2")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Help_functions_v2 import sklearn_Pvalue, sklearn_adjR2, RMSE
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#1: import data
train_data=pd.read_csv("C:/Users/Haixiang He/Desktop/仲益\Kaggle/data/train_clean2.csv")
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


#2.1 optimized by looking at the learning curve
alphaList = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 20, 30, 40, 50, 60]
trainRmseList=[]
testRmseList=[]

for i in alphaList:
    #fit model for each alpha and calcualte train and test set rmse
    ridge_model=Ridge(alpha=i)
    ridge_model.fit(X_train, y_train)
    y_trainPred=ridge_model.predict(X_train)
    trainRmseList.append(RMSE(y_trainPred,y_train))
    y_testPred=ridge_model.predict(X_test)
    testRmseList.append(RMSE(y_testPred,y_test))
    
#plot train and test sets rmse
data = pd.DataFrame([alphaList, trainRmseList,testRmseList ])
data=data.transpose()
data.columns=["alpha","train set rmse", "test set rmse"]
data.plot.scatter(x="alpha", y='train set rmse');
data.plot.scatter(x="alpha", y='test set rmse');

#2.2.1 optimized ridge model parameter using cross-validation
#cross validate alpha using 5-fold
ridgeModel = RidgeCV(alphas = alphaList, cv=5)
ridgeModel.fit(train_data, dependentV)
#get the best alpha
alphaBest = ridgeModel.alpha_
print("Best alpha for ridge model is ", alphaBest)
#try to find a more precised alpha
print("Try again for more precision around " + str(alphaBest))
temp=[0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3]
ridgeModel = RidgeCV(alphas = [alphaBest*i for i in temp], cv = 5)
ridgeModel.fit(train_data, dependentV)
#find the more precise alpha
alphaBest = ridgeModel.alpha_
#look at the result 
print("Best alpha for ridge model is ", alphaBest)
y_pred=ridgeModel.predict(train_data)
print("RMSE for ridge model is ", RMSE(y_pred,dependentV))
# Plot important coefficients
coefs = pd.Series(ridgeModel.coef_, index = train_data.columns)
print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Ridge Model")
plt.show()

#2.2.2 optimized lasso model parameter using cross-validation
#cross validate alpha using 5-fold
lassoModel = LassoCV(alphas = alphaList, cv=5)
lassoModel.fit(train_data, dependentV)
#get the best alpha
alphaBest = lassoModel.alpha_
print("Best alpha for lasso model is ", alphaBest)
#try to find a more precised alpha
print("Try again for more precision around " + str(alphaBest))
temp=[0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3]
lassoModel = LassoCV(alphas = [alphaBest*i for i in temp], cv = 5)
lassoModel.fit(train_data, dependentV)
#find the more precise alpha
alphaBest = lassoModel.alpha_
#look at the result 
print("Best alpha for lasso model is ", alphaBest)
y_pred=lassoModel.predict(train_data)
print("RMSE for lasso model is ", RMSE(y_pred,dependentV))
# Plot important coefficients
coefs = pd.Series(lassoModel.coef_, index = train_data.columns)
print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
      str(sum(coefs == 0)) + " features")
imp_coefs = pd.concat([coefs.sort_values().head(10),
                     coefs.sort_values().tail(10)])
imp_coefs.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()


#3 PCA
#standardized before PCA:
#minus mean and divide by standard deviation
sc = StandardScaler()
train_data = sc.fit_transform(train_data)
#run pca
pcaDecompose = PCA()
pcaDecompose.fit(train_data)
#explantion of variance for each PCs
#eigenvalue
variance = pd.DataFrame(pcaDecompose.explained_variance_ratio_)
print (variance)
#find the best n PCs explain majority (for example 90%) of the variance
cumVariance=np.cumsum(pcaDecompose.explained_variance_ratio_)
for i in range(len(cumVariance)):
    if cumVariance[i]>0.9:
        n_components=i
        break
#reduce train_data ot 87 columns
pca = PCA(n_components=n_components)
pca = pca.fit(train_data)
dataPCA = pd.DataFrame(pca.transform(train_data))
print(dataPCA.shape)

#homework, run multivariate regression with the PCA result






