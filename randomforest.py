# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:42:17 2021

@author: SRT
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
data  = pd.read_csv("C:/Users/SRT/Downloads/diabetes.csv")
data
x = data.drop(["Outcome","BloodPressure","SkinThickness"],axis =1)
y = data["Outcome"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3,random_state = 0)
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)
z = model.score(x_test,y_test)
predictes = model.predict(x_test)
cm = confusion_matrix(y_test,predictes)
cm