# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 13:36:23 2022

@author: ramav
"""
#SIMPLE LINEAR REGREESSION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("D:\\Assignments\\simple linear regresssion\\salary_data.csv")

df.head()
df.shape

df.isnull().sum()


x = df.iloc[:,:1].values
y = df.iloc[:,-1].values

#EDA
plt.scatter(x, y, color="red")
plt.title(" relation between salary and experience")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

plt.boxplot(x)
plt.show()
plt.hist(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)
x_train.shape


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_train, y_train)

#knowing Bo(inrecept) and B1 value(coefficient)
LR.intercept_.round(3) #Bo
LR.coef_.round(3) #B1


y_pred_train = LR.predict(x_train)
y_pred_test = LR.predict(x_test)

y_pred_train
y_pred_test

# calculating mean square eror and Root of mean square error
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y_train,y_pred_train)

RMSE = np.sqrt(mse)
print("Root mean square :", RMSE.round(2)) #RMSE=5415.91

print("R square:",r2_score(y_train,y_pred_train).round(2)*100) #96


mse1 = mean_squared_error(y_test,y_pred_test)
RMSE1= np.sqrt(mse1)
print("Root mean square :", RMSE1.round(2)) #RMSE=5415.91

print("R square:",r2_score(y_test,y_pred_test).round(2)*100)



import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,y_pred_train,color="red")
plt.title("training scatter plot")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_test,y_pred_test,color="red")
plt.title("test scatter plot")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("D:\\Assignments\\simple linear regresssion\\salary_data.csv")

df.head()
df.shape

# x and y variable
x = df["YearsExperience"]
y = df["Salary"]


#split as train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=42)

x_train.shape

#scatter plot between x and y
df.plot(kind="scatter",x="YearsExperience",y="Salary")
plt.show()

import matplotlib.pyplot as plt
plt.scatter(x,y,color="red",edgecolors="orange")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

#box plot to know outliers
df.plot(kind="box")
plt.show()

df.corr()

# Dataframe
x_train = pd.DataFrame(x_train)
y_train= pd.DataFrame(y_test)

x_ test= pd.DataFrame(x_test)
y_test= pd.DataFrame(y_test)



# model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_train,y_train)

#knowing Bo(inrecept) and B1 value(coefficient)

LR.intercept_.round(3)
LR.coef_.round(3)

LR.score(x,y).round(3)

#prediction
y_pred_train = LR.predict(x_train)
y_pred_train

y_pred_test = LR.predict(x_test)
y_pred_test

#comparsion between y actual and y_pred by using scatter plot
plt.scatter(x=x.iloc[:,0],y=y,color="red")
plt.plot(x.iloc[:,0], y_pred_train,color="blue")
plt.xlabel("YearsExperience")
plt.ylabel("salary")
plt.show()

# calculating mean square eror and Root of mean square error
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(y,y_pred_train)

RMSE = np.sqrt(mse)
print("Root mean square :", RMSE.round(2))

print("R square:",r2_score(y,y_pred).round(2)*100)

'''
