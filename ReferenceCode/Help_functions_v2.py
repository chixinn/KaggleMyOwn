##COPYRIGHT from 仲益教育
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import f_regression
#from scipy import stats

#calcualte RMSE -- root mean squared error
def RMSE (y_pred, y_actual):
    return np.sqrt(mean_squared_error(y_actual,y_pred))

#calualte p value
def sklearn_Pvalue(X, y):
    lm = LinearRegression()
    lm.fit(X,y)
    F,pval=f_regression(X, y, False)
    myDF3 = pd.DataFrame()
    myDF3["Coefficients"],myDF3["t values"],myDF3["Probabilities"] = [np.array(lm.coef_),np.array(np.sqrt(F)),np.array(pval)]
    return(myDF3)

#calculate adjusted R^2
def sklearn_adjR2(X, y):
    lr=LinearRegression()
    lr.fit(X, y)
    y_trainPred=lr.predict(X)
    r2=r2_score(y, y_trainPred)
    p=len(X.columns)+1
    n=len(X)
    return 1-(1-r2)*(n-1)/(n-p-1)
    


    
    
    
    
    