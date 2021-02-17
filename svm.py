# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:36:41 2021

@author: SRT
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
data  = pd.read_csv("C:/Users/SRT/Downloads/diabetes.csv")
data
x = data.drop(["Outcome","BloodPressure","SkinThickness"],axis =1)
y = data["Outcome"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3,random_state = 0)
cls = svm.SVC(kernel="linear")
cls.fit(x_train,y_train)
pred = cls.predict(x_test)
z = metrics.accuracy_score(y_test,pred)
z
cm = confusion_matrix(y_test,pred)
cm
