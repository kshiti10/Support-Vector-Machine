# -*- coding: utf-8 -*-
"""
Created on Mar 13 07:45:37 2024

@author: Kshitija
"""

#       SVM 
'''
In California, annual forest fires can cause huge loss of wildlife,
 human life, and can cost billions of dollars in property damage.
 Local officials would like to predict the size of the burnt area in
 forest fires annually so that they can be better prepared in future calamities. 
Build a Support Vector Machines algorithm on the dataset and share your insights on it in the documentation. 
'''


#Dataset
#month             month of the year: "jan" to "dec"
#day               day of the week: "mon" to "sun"
#FFMC             FFMC index from the FWI system: 18.7 to 96.20
#DMC              DMC index from the FWI system: 1.1 to 291.3
#DC               DC index from the FWI system: 7.9 to 860.6
#ISI              ISI index from the FWI system: 0.0 to 56.10
#temp             temperature in Celsius degrees: 2.2 to 33.30
#RH                 relative humidity in %: 15.0 to 100
#wind             wind speed in km/h: 0.40 to 9.40
#rain             outside rain in mm/m2 : 0.0 to 6.4
#area             the burned area of the forest (in ha): 0.00 to 1090.84 
#dayfri             int64
#daymon             int64
#daysat             int64
#daysun             int64
#daythu             int64
#daytue             int64
#daywed             int64
#monthapr           int64
#monthaug           int64
#monthdec           int64
#monthfeb           int64
#monthjan           int64
#monthjul           int64
#monthjun           int64
#monthmar           int64
#monthmay           int64
#monthnov           int64
#monthoct           int64
#monthsep           int64
#size_category     object
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
forest=pd.read_csv("c:/datasets/forestfires.csv")


forest.dtypes

################
###EDA
forest.shape

plt.figure(1,figsize=(16,10))
sns.countplot(forest.month)
#Aug and sept has highest value
sns.countplot(forest.day)
#Friday sunday and saturday has highest value

sns.distplot(forest.FFMC)
#data isnormal and slight left skewed
sns.boxplot(forest.FFMC)
#There are several outliers

sns.distplot(forest.DMC)
#data isnormal and slight right skewed
sns.boxplot(forest.DMC)
#There are several outliers

sns.distplot(forest.DC)
#data isnormal and slight left skewed
sns.boxplot(forest.DC)
#There are  outliers

sns.distplot(forest.ISI)
#data isnormal 
sns.boxplot(forest.ISI)
#There are  outliers

sns.distplot(forest.temp)
#data isnormal a
sns.boxplot(forest.temp)
#There are  outliers

sns.distplot(forest.RH)
#data isnormal and slight left skewed
sns.boxplot(forest.RH)
#There are  outliers

sns.distplot(forest.wind)
#data isnormal and slight right skewed
sns.boxplot(forest.wind)
#There are  outliers

sns.distplot(forest.rain)
#data isnormal 
sns.boxplot(forest.rain)
#There are  outliers

sns.distplot(forest.area)
#data isnormal 
sns.boxplot(forest.area)
#There are  outliers

#Now let us check the Highest Fire In KM?
forest.sort_values(by="area", ascending=False).head(5)

highest_fire_area = forest.sort_values(by="area", ascending=True)

plt.figure(figsize=(8, 6))

plt.title("Temperature vs area of fire" )
plt.bar(highest_fire_area['temp'], highest_fire_area['area'])

plt.xlabel("Temperature")
plt.ylabel("Area per km-sq")
plt.show()
#once the fire starts,almost 1000+ sq area's temperature goes beyond 25 and 
#around 750km area is facing temp 30+
#Now let us check the highest rain in the forest
highest_rain = forest.sort_values(by='rain', ascending=False)[['month', 'day', 'rain']].head(5)
highest_rain
#highest rain observed in the month of aug
#Let us check highest and lowest temperature in month and day wise
highest_temp = forest.sort_values(by='temp', ascending=False)[['month', 'day', 'temp']].head(5)

lowest_temp =  forest.sort_values(by='temp', ascending=True)[['month', 'day', 'temp']].head(5)

print("Highest Temperature",highest_temp)
#Highest temp observed in aug
print("Lowest_temp",lowest_temp)
#lowest temperature in the month of dec

forest.isna().sum()
#There no missing values in both

################################
sal1.dtypes

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
forest.month=labelencoder.fit_transform(forest.month)
forest.day=labelencoder.fit_transform(forest.day)
forest.size_category=labelencoder.fit_transform(forest.size_category)

forest.dtypes
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['month'])
df_t=winsor.fit_transform(forest[["month"]])
sns.boxplot(df_t.month)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['FFMC'])
df_t=winsor.fit_transform(forest[["FFMC"]])
sns.boxplot(df_t.FFMC)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['DMC'])
df_t=winsor.fit_transform(forest[["DMC"]])
sns.boxplot(df_t.DMC)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['DC'])
df_t=winsor.fit_transform(forest[["DC"]])
sns.boxplot(df_t.DC)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['ISI'])
df_t=winsor.fit_transform(forest[["ISI"]])
sns.boxplot(df_t.ISI)


from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['temp'])
df_t=winsor.fit_transform(forest[["temp"]])
sns.boxplot(df_t.temp)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['RH'])
df_t=winsor.fit_transform(forest[["RH"]])
sns.boxplot(df_t.RH)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['wind'])
df_t=winsor.fit_transform(forest[["wind"]])
sns.boxplot(df_t.wind)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['rain'])
df_t=winsor.fit_transform(forest[["rain"]])
sns.boxplot(df_t.rain)

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['area'])
df_t=winsor.fit_transform(forest[["area"]])
sns.boxplot(df_t.area)
###########################################

tc = forest.corr()
tc
fig,ax= plt.subplots()
fig.set_size_inches(200,10)
sns.heatmap(tc, annot=True, cmap='YlGnBu')
#all the variables are moderately correlated with size_category except area

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test=train_test_split(forest,test_size=0.3)
train_X=train.iloc[:,:30]
train_y=train.iloc[:,30]
test_X=test.iloc[:,:30]
test_y=test.iloc[:,30]
#Kernel linear
model_linear=SVC(kernel="linear")
model_linear.fit(train_X,train_y)
pred_test_linear=model_linear.predict(test_X)
np.mean(pred_test_linear==test_y)
#RBF
model_rbf=SVC(kernel="rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf=model_rbf.predict(test_X)
np.mean(pred_test_rbf==test_y)
