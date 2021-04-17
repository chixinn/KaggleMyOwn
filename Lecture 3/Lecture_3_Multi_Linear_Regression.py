##COPYRIGHT from 仲益教育
import sys
sys.path.append("C://Users//Haixiang He//desktop//Kaggle//Lecture 2")
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from Help_functions_v2 import sklearn_Pvalue, sklearn_adjR2, RMSE
import numpy as np

#1: import data
train_data=pd.read_csv("C:/Users/Haixiang He/Desktop/仲益\Kaggle/data/train_clean.csv")
#2: feature engineering
#2.1: take a closer look at the categorical features before changing to dummy
categorical_features = train_data.select_dtypes(include = ["object"]).columns
train_cat = train_data[categorical_features]
#see what most category data are the same
#delete highly skew category data
pct=[]
for ix in train_cat.columns:
    temp=train_cat[ix].describe()
    pct.append(temp["freq"]/temp["count"])
skewData=pd.DataFrame(pct,index=train_cat.columns,columns=["skewness"])
skewData=skewData.sort_values(by="skewness",ascending=False)
print (skewData)
train_data = train_data.drop((skewData[skewData['skewness'] >= 0.95]).index,1) 

#2.2: some categorical features when there is information in the order
#Alley: Type of alley access to property
train_data=train_data.replace({"Alley":{"Grvl" : 1, "Pave" : 2}})
#BsmtCond: Evaluates the general condition of the basement
train_data=train_data.replace({"BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})
#BsmtExposure: Refers to walkout or garden level walls
train_data=train_data.replace({"BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3}})
#BsmtFinType1: Rating of basement finished area
train_data=train_data.replace({"BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6}})
#BsmtFinType2: Rating of basement finished area (if multiple types)
train_data=train_data.replace({"BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6}})
#BsmtQual: Evaluates the height of the basement
train_data=train_data.replace({"BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5}})
#ExterCond: Evaluates the present condition of the material on the exterior
train_data=train_data.replace({"ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5}})
#ExterQual: Evaluates the quality of the material on the exterior 
train_data=train_data.replace({"ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5}})
#FireplaceQu: Fireplace quality
train_data=train_data.replace({"FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})
#Functional: Home functionality (Assume typical unless deductions are warranted)
train_data=train_data.replace({"Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8}})
#HeatingQC: Heating quality and condition
train_data=train_data.replace({"HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})
#KitchenQual: Kitchen quality
train_data=train_data.replace({"KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5}})
#LandSlope: Slope of property
train_data=train_data.replace({"LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3}})
#LotShape: General shape of property
train_data=train_data.replace({"LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4}})
#PavedDrive: Paved driveway
train_data=train_data.replace({"PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2}})

#2.3: Simplifications of existing categorical 
train_data["Condition1"] = train_data.Condition1.replace({"RRNe" : "Other", 
                                                  "RRNn" : "Other","PosA" : "Other", 
                                                   "RRAe" : "Other"
                                                  })
train_data["Electrical"] = train_data.Electrical.replace({"Mix" : "Other", 
                                                  "FuseP" : "Other"
                                                  })
train_data["Exterior1st"] = train_data.Exterior1st.replace({"AsphShn" : "Other", 
                                                  "CBlock" : "Other","ImStucc" : "Other", 
                                                   "BrkComm" : "Other","Stone" : "Other"
                                                  })
train_data["Exterior2nd"] = train_data.Exterior2nd.replace({"CBlock" : "Other", 
                                                  "AsphShn" : "Other","Stone" : "Other", 
                                                   "Brk Cmn" : "Other","ImStucc" : "Other"
                                                  })
train_data["Foundation"] = train_data.Foundation.replace({"Wood" : "Other", 
                                                  "Stone" : "Other"
                                                  })
train_data["GarageType"] = train_data.GarageType.replace({"2Types" : "Other", 
                                                  "CarPort" : "Other"
                                                  })
train_data["HouseStyle"] = train_data.HouseStyle.replace({"2.5Fin" : "Other", 
                                                  "2.5Unf" : "Other",
                                                  "1.5Unf" : "Other"
                                                  })
train_data["LotConfig"] = train_data.LotConfig.replace({"FR3" : "FR2"
                                                  })
train_data["MSSubClass"] = train_data.MSSubClass.replace({"SC40" : "Other", 
                                                  "SC180" : "Other",
                                                  "SC45" : "Other",
                                                  "SC75" : "Other"
                                                  })
train_data["MSZoning"] = train_data.MSZoning.replace({"C (all)" : "Other", 
                                                  "RH" : "Other"
                                                  })
train_data["Neighborhood"] = train_data.Neighborhood.replace({"Blueste" : "Other", 
                                                  "NPkVill" : "Other",
                                                  "Veenker" : "Other"
                                                  })
train_data["RoofStyle"] = train_data.RoofStyle.replace({"Shed" : "Other", 
                                                  "Mansard" : "Other",
                                                  "Gambrel" : "Other",
                                                  "Flat" : "Other"
                                                  })
train_data["SaleCondition"] = train_data.SaleCondition.replace({"AdjLand" : "Other", 
                                                  "Alloca" : "Other"
                                                  })
train_data["SaleType"] = train_data.SaleType.replace({"Con" : "Other", 
                                                  "Oth" : "Other",
                                                  "CWD" : "Other", "ConLI" : "Other",
                                                  "ConLw" : "Other","ConLD" : "Other"
                                                  })

#2.4 Combinations of existing features
# Overall quality of the house
train_data["OverallGrade"] = train_data["OverallQual"] * train_data["OverallCond"]
# Overall quality of the exterior
train_data["ExterGrade"] = train_data["ExterQual"] * train_data["ExterCond"]
# Overall kitchen score
train_data["KitchenScore"] = train_data["KitchenAbvGr"] * train_data["KitchenQual"]
# Total number of bathrooms
train_data["TotalBath"] = train_data["BsmtFullBath"] + (0.5 * train_data["BsmtHalfBath"]) + \
train_data["FullBath"] + (0.5 * train_data["HalfBath"])
train_data.drop(["BsmtFullBath","BsmtHalfBath","FullBath","HalfBath"],axis=1)
# Total SF for house (incl. basement)
train_data["AllSF"] = train_data["GrLivArea"] + train_data["TotalBsmtSF"]
train_data.drop(["GrLivArea","TotalBsmtSF"],axis=1)
# Total SF for 1st + 2nd floors
train_data["AllFlrsSF"] = train_data["1stFlrSF"] + train_data["2ndFlrSF"]
train_data.drop(["1stFlrSF","2ndFlrSF"],axis=1)
# Total SF for porch
train_data["AllPorchSF"] = train_data["OpenPorchSF"] + train_data["EnclosedPorch"] + \
train_data["3SsnPorch"] + train_data["ScreenPorch"]
train_data.drop(["OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch"],axis=1)
# House completed before sale or not
train_data["BoughtOffPlan"] = train_data.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                      "Family" : 0, "Normal" : 0, "Partial" : 1})

#2.5 Polinomial transformation (Box-Cox)
# X^2, X^3, X^0.5, 1/X, Log(X)
# Find most important features relative to target
print("Find most important features relative to target")
corr = train_data.corr()
corr.sort_values(["SalePrice"], ascending = False, inplace = True)
print(corr.SalePrice)
train_data["OverallQual-s2"] = train_data["OverallQual"] ** 2
train_data["AllSF-2"] = train_data["AllSF"] ** 2
train_data["AllFlrsSF-2"] = train_data["AllFlrsSF"] ** 2
train_data["GrLivArea-2"] = train_data["GrLivArea"] ** 2
# train_data["SimplOverallQual-s2"] = train_data["SimplOverallQual"] ** 2
train_data["ExterQual-2"] = train_data["ExterQual"] ** 2
train_data["GarageCars-2"] = train_data["GarageCars"] ** 2
train_data["TotalBath-2"] = train_data["TotalBath"] ** 2
train_data["KitchenQual-2"] = train_data["KitchenQual"] ** 2

# Differentiate numerical features and categorical features
categorical_features = train_data.select_dtypes(include = ["object"]).columns
numerical_features = train_data.select_dtypes(exclude = ["object"]).columns
numerical_features = numerical_features.drop("SalePrice")
print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
train_num = train_data[numerical_features]
train_cat = train_data[categorical_features]
print("NAs for numerical features in train : " + str(train_num.isnull().values.sum()))
train_num = train_num.fillna(train_num.median())
print("Remaining NAs for numerical features in train : " + str(train_num.isnull().values.sum()))

#2.6 turn category features to dummy
print("NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))
train_cat = pd.get_dummies(train_cat, drop_first=True)
print("Remaining NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))

#2.7 join categorical and numerical features 
train_Data_New = pd.concat([train_num, train_cat], axis = 1)
print("New number of features : " + str(train_Data_New.shape[1]))

#2.8 remove collinear columns
# Create correlation matrix
corr_matrix = train_Data_New.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
# Drop features 
train_Data_New.drop(to_drop, axis=1, inplace=True)


#3.split in-sample data to training and validation sets
train_Data_New.index=train_Data_New["Id"]
train_Data_New.drop("Id", axis=1, inplace=True)

train_data.index=train_data["Id"]
dependentV=train_data["SalePrice"]
#Partition the dataset in train + validation sets
#usually linear regression needs at least 30 observations
#split of train and validation can be 70:30, or 60:40
X_train, X_test, y_train, y_test = train_test_split(train_Data_New, dependentV, test_size = 0.3, random_state = 0)
print("X_train : " + str(X_train.shape))
print("X_test : " + str(X_test.shape))
print("y_train : " + str(y_train.shape))
print("y_test : " + str(y_test.shape))

#3.1 build multi linear regression 
lr=LinearRegression()
lr.fit(X_train, y_train)
y_trainPred=lr.predict(X_train)
#look at r_squared, adjusted r_squared
print("r-squared of in-sample is")
r2_score(y_train, y_trainPred)
adj_r2=sklearn_adjR2(X_train, y_train)
print("adjusted r-squared of in-sample is")
print(adj_r2)
print("RMSE of in-sample is")
print(RMSE(y_trainPred,y_train))
print(sklearn_Pvalue(X_train, y_train))

# 3.2 out of sample
y_testPred=lr.predict(X_test)
#look at  r_squared, adjusted r_squared
print("r-squared of out-of-sample is")
r2_score(y_test, y_testPred)
print("adjusted r-squared of out-of-sample is")
adj_r2=sklearn_adjR2(X_test, y_test)
print("RMSE of out-of-sample is")
print(RMSE(y_testPred, y_test))
print(sklearn_Pvalue(X_test, y_test))

train_Data_New["SalePrice"]=train_data["SalePrice"]
#4
#export data
train_Data_New.to_csv("C:/Users/Haixiang He/Desktop/Kaggle/data/train_clean2.csv")