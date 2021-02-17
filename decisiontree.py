# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as pnp
import seaborn as sns 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#will load files

data  = pd.read_csv("C:/Users/SRT/Downloads/diabetes.csv")
data
x = data.drop(["Outcome","BloodPressure","SkinThickness"],axis =1)
y = data["Outcome"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3,random_state = 0)
clf_entropy = DecisionTreeClassifier(criterion="entropy",random_state=0,max_depth=3,min_samples_leaf=5)
clf_entropy.fit(x_train,y_train)
y_prediction = clf_entropy.predict(x_test)
y_prediction
score = accuracy_score( y_test, y_prediction)
score
sns.heatmap(data.corr())