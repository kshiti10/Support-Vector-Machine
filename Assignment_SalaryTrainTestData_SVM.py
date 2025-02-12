# -*- coding: utf-8 -*-
"""
Created Mar 13 07:53:41 2024

@author: Kshitija
"""
'''
A construction firm wants to develop a suburban locality with 
new infrastructure but they might incur losses if they cannot 
sell the properties. To overcome this, they consult an analytics 
firm to get insights on how densely the area is populated and 
the income levels of residents. Use the Support Vector Machines 
algorithm on the given dataset and draw out insights and also
 comment on the viability of investing in that area.
 '''
 
 
 #Dataset
#age               int64
#workclass        object
#education        object
#educationno       int64
#maritalstatus    object
#occupation       object
#relationship     object
#race             object
#sex              object
#capitalgain       int64
#capitalloss       int64
#hoursperweek      int64
#native           object
#Salary           object
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sal1=pd.read_csv("C:/datasets/SalaryData_Train.csv")
sal2=pd.read_csv("C:/datasets/SalaryData_Test.csv")

sal1.dtypes
sal2.dtypes
################
###EDA
#age is continious variable
import seaborn as sns
sns.distplot(sal1.age)
#Age is right skewed
sns.boxplot(sal1.age)
#There are several outliers
sal1.isna().sum()
sal2.isna().sum()
#There no missing values in both

plt.figure(1,figsize=(16,10))
sns.countplot(sal1.workclass)
#There are higher number of private employees

# now let us plot count plot
plt.figure(1,figsize=(16,10))
sns.countplot(sal1.education)
#There are very high number of HSC 

#educationno  is equivalent to education hence can be dropped
sal1=sal1.drop(["educationno"],axis=1)
sal2=sal2.drop(["educationno"],axis=1)
#marital-status
# now let us plot count plot
plt.figure(1,figsize=(16,10))
sns.countplot(sal1.maritalstatus)
#There are married-civ-spouse highest in number
#occupation
plt.figure(1,figsize=(16,10))
sns.countplot(sal1.occupation)
#professor speciality and ex-manger are higher in number
#relationship
plt.figure(1,figsize=(16,10))
sns.countplot(sal1.relationship)
#Working husbands are more
#race
plt.figure(1,figsize=(16,10))
sns.countplot(sal1.race)
#White people are more in this coloney
plt.figure(1,figsize=(16,10))
sns.countplot(sal1.sex)
#More number of males are staying in this coloney
#capital-gain and capital-loss are irrelevant hence can be dropped
sal1=sal1.drop(["capitalgain"],axis=1)
sal1=sal1.drop(["capitalloss"],axis=1)
sal2=sal2.drop(["capitalgain"],axis=1)
sal2=sal2.drop(["capitalloss"],axis=1)
#hours-per-week
sns.distplot(sal1.hoursperweek)
#hoursperweek  is normal
sns.boxplot(sal1.hoursperweek)
#There are several outliers
#native-country
plt.figure(1,figsize=(16,10))
sns.countplot(sal1.native)
#majority are US native very few are from other country
#Salary ,this is response variable
plt.figure(1,figsize=(16,10))
sns.countplot(sal1.Salary)
#people having salary less than 50k are more in this coloney
################################
sal1.dtypes

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
sal1.workclass=labelencoder.fit_transform(sal1.workclass)
sal1.education=labelencoder.fit_transform(sal1.education)
sal1.maritalstatus=labelencoder.fit_transform(sal1.maritalstatus)
sal1.occupation=labelencoder.fit_transform(sal1.occupation)
sal1.relationship=labelencoder.fit_transform(sal1.relationship)
sal1.race=labelencoder.fit_transform(sal1.race)
sal1.sex=labelencoder.fit_transform(sal1.sex )
sal1.native=labelencoder.fit_transform(sal1.native)
sal1.Salary=labelencoder.fit_transform(sal1.Salary)

sal2.workclass=labelencoder.fit_transform(sal2.workclass)
sal2.education=labelencoder.fit_transform(sal2.education)
sal2.maritalstatus=labelencoder.fit_transform(sal2.maritalstatus)
sal2.occupation=labelencoder.fit_transform(sal2.occupation)
sal2.relationship=labelencoder.fit_transform(sal2.relationship)
sal2.race=labelencoder.fit_transform(sal2.race)
sal2.sex=labelencoder.fit_transform(sal2.sex )
sal2.native=labelencoder.fit_transform(sal2.native)
sal2.Salary=labelencoder.fit_transform(sal2.Salary)
#
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['age'])
df_t=winsor.fit_transform(sal1[["age"]])
sns.boxplot(df_t.age)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['hoursperweek'])
df_t=winsor.fit_transform(sal1[["hoursperweek"]])
sns.boxplot(df_t.hoursperweek)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['age'])
df_t=winsor.fit_transform(sal2[["age"]])
sns.boxplot(df_t.age)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['hoursperweek'])
df_t=winsor.fit_transform(sal2[["hoursperweek"]])
sns.boxplot(df_t.hoursperweek)

tc = sal1.corr()
tc
fig,ax= plt.subplots()
fig.set_size_inches(200,10)
sns.heatmap(tc, annot=True, cmap='YlGnBu')
#salary,age,sex and hoursperweek are highly correlated

from sklearn.svm import SVC
train_X=sal1.iloc[:,:10]
train_y=sal1.iloc[:,10]
test_X=sal2.iloc[:,:10]
test_y=sal2.iloc[:,10]
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
