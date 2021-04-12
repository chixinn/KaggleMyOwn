##COPYRIGHT from 仲益教育

##Part 1: Data Exploration
#0.0: import packages
#pandas is a data structure package
#seaborn is a statistial data visualization
#matplotlib is a plotting package
#numpy is a scientific calculation in Python
#sklearn is a data science package
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from scipy import stats

#0.1: import data
#change to your local path
train_data=pd.read_csv("/Users/chixinning/Desktop/kaggle/KaggleMyOwn/Lecture_0/data/train.csv")

#1.1: look at what's in the data
train_data.columns
#look at data description, look at variable table
#or use excel to look at the data
#or use the following useful functions
train_data.shape
train_data.head()
train_data.tail()
#Change dataframe index to Id
# 通过房子的features预测房价 
train_data.index=train_data["Id"]
train_data.drop("Id", axis=1, inplace=True)

#2.1: look at what do we need to predict first
#decriptive summary, look at the distribution of data
train_data["SalePrice"].describe()
# 通过describe 去看fat-tail
#skewness and kurtosis
print("Skewness: %f" % train_data['SalePrice'].skew())
print("Kurtosis: %f" % train_data['SalePrice'].kurt())
#histogram
sns.distplot(train_data['SalePrice'])

#2.2: look at what are the variables
#select some variables you think are relevant
#2.2.1 numerical variables: try LotArea (lot size in square feet) and year built
#scatter plot
var = 'LotArea'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
var = 'YearBuilt'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#2.2.2 categorical variables: try OverQual (Overall material and finish quality)
#boxplot
var = 'OverallQual'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
#2.2.3 correlation
#heat map
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
plt.show()
#scatter plot 
# 不太懂画这个可视化可以看出来啥
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF']
sns.pairplot(train_data[cols], height = 2.5)
plt.show();

#3.1 data cleaning: no correct answer
# how many % of data are missing
total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
print("how many data has missing value = " + str(sum(missing_data['Total'] >= 1)))
#3.1.1 fill missing value with mode
train_data["Electrical"]=train_data["Electrical"].fillna(train_data["Electrical"].mode()[0])
#3.1.2 fill missing value with 0

#3.1.5 #dealing with missing data
#need to understand
# 这里的缺失值直接drop了？ 
# 这里面没有就是没有的部分 用None处理缺失值保证没有，然后后面categorial 向numerical转换的时候，转换成程度最低的数值。
train_data["PoolQC"]=train_data["PoolQC"].fillna("None")
train_data["Alley"]=train_data["Alley"].fillna("None")
train_data["Fence"]=train_data["Fence"].fillna("None")
train_data["FireplaceQu"]=train_data["FireplaceQu"].fillna("None ")
train_data["GarageCond"]=train_data["GarageCond"].fillna("None")
train_data["GarageType"]=train_data["GarageType"].fillna("None ")
train_data["GarageYrBlt"]=train_data["GarageYrBlt"].fillna("None ")
train_data["GarageFinish"]=train_data["GarageFinish"].fillna("None ")
train_data["GarageQual"]=train_data["GarageQual"].fillna("None ")
train_data["BsmtExposure"]=train_data["BsmtExposure"].fillna("None ")
train_data["BsmtExposure"]=train_data["BsmtExposure"].fillna("None ")
train_data["BsmtFinType2"]=train_data["BsmtFinType2"].fillna("None ")
train_data["BsmtFinType1"]=train_data["BsmtFinType1"].fillna("None ")
train_data["BsmtCond"]=train_data["BsmtCond"].fillna("None ")
train_data["BsmtQual"]=train_data["BsmtQual"].fillna("None ")
train_data["MasVnrType"]=train_data["MasVnrType"].fillna("None ")


# # 数值型的用median填补
# 比较mode 和 median在缺失值填补重的效果：看分布是否偏态? 所以众数和中位数的区别
train_data["LotFrontage"]=train_data["LotFrontage"].fillna(train_data["LotFrontage"].median())
train_data["LotFrontage"]=train_data["LotFrontage"].fillna(train_data["LotFrontage"].mode()[0])

# train_data["MasVnrArea"]=train_data["MasVnrArea"].fillna(0)
#3.1.3 OR fill missing value with None
# train_data["MasVnrArea"]=train_data["MasVnrArea"].fillna("None")
#3.1.4 OR fill missing value with median
train_data["MasVnrArea"]=train_data["MasVnrArea"].fillna(train_data["MasVnrArea"].median())
train_data["MasVnrArea"]=train_data["MasVnrArea"].fillna(train_data["MasVnrArea"].mode()[0])

# 存疑
train_data["Electrical"]=train_data["Electrical"].fillna(train_data["Electrical"].mode()[0])







total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
#缺失值最好不要全部删掉
train_data = train_data.drop((missing_data[missing_data['Total'] >= 1]).index,1)
#just checking that there's no missing data missing
train_data.isnull().sum().max() 

#3.2 outliers
#z-score
saleprice_scaled = StandardScaler().fit_transform(train_data['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
#scatter chart
var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#3.2.1 deleting points
train_data.sort_values(by = 'GrLivArea', ascending = False)[:2]
# train_data = train_data.drop(1299)
# train_data = train_data.drop(524)
var = 'TotalBsmtSF'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

#3.2.2 log transformation
#histogram and normal probability plot
print("Skewness: %f" % train_data['SalePrice'].skew())
print("Kurtosis: %f" % train_data['SalePrice'].kurt())
sns.distplot(train_data['SalePrice'], fit=norm);
fig = plt.figure()
# 找出skew 很大的变量
print("Skewness: %f" % train_data['LotFrontage'].skew()) # Skewness: 2.409147

print("Skewness: %f" % train_data['LotArea'].skew()) # Skewness: 12.207688

print("Skewness: %f" % train_data['BsmtFinSF1'].skew()) # Skewness: 1.685503
    
print("Skewness: %f" % train_data['BsmtFinSF2'].skew()) # Skewness: 1.685503

print("Skewness: %f" % train_data['BsmtUnfSF'].skew()) # Skewness: 1.685503
print("Skewness: %f" % train_data['TotalBsmtSF'].skew()) # Skewness: 1.685503

print("Skewness: %f" % train_data['1stFlrSF'].skew()) # Skewness: 1.685503
print("Skewness: %f" % train_data['2ndFlrSF'].skew()) # Skewness: 1.685503

print("Skewness: %f" % train_data['LowQualFinSF'].skew()) # Skewness: 1.685503
print("Skewness: %f" % train_data['GrLivArea'].skew()) # Skewness: 1.685503
print("Skewness: %f" % train_data["Fireplaces"].skew()) # Skewness: 1.685503
print("Skewness: %f" % train_data["GarageArea"].skew()) # Skewness: 1.685503


print("Skewness: %f" % train_data['WoodDeckSF'].skew()) # Skewness: 1.685503
print("Skewness: %f" % train_data['OpenPorchSF'].skew()) # Skewness: 1.685503
print("Skewness: %f" % train_data["EnclosedPorch"].skew()) # Skewness: 1.685503
print("Skewness: %f" % train_data["3SsnPorch"].skew()) 

print("Skewness: %f" % train_data["ScreenPorch"].skew()) # Skewness: 1.685503
print("Skewness: %f" % train_data["PoolArea"].skew()) 

#qq plot
res = stats.probplot(train_data['SalePrice'], plot=plt)
#applying log transformation
train_data['SalePrice'] = np.log(train_data['SalePrice'])
print("Skewness: %f" % train_data['SalePrice'].skew())
print("Kurtosis: %f" % train_data['SalePrice'].kurt())
sns.distplot(train_data['SalePrice'], fit=norm);
fig = plt.figure()
#qq plot
res = stats.probplot(train_data['SalePrice'], plot=plt)

#3.3 convert numerical variables to categorical variables
train_data = train_data.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })
# categorial -> numerical


#export data
train_data.to_csv("train_clean.csv")