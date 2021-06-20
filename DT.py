# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 00:41:21 2021

@author: Mi8a
"""

import numpy as nm  
import matplotlib.pyplot as plt  
import pandas as pd  
  

#importing datasets  
dataset= pd.read_csv('spambase.csv')  
  
dataset.shape

dataset.head()

X = dataset.drop('class', axis=1)
y = dataset['class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.tree import DecisionTreeClassifier, plot_tree
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Tree Visualisation

fig = plt.figure(figsize=(100,80))
plot = plot_tree(classifier, feature_names=list(dataset.columns), class_names=['edible', 'poisonous'],filled=True)
for i in plot:
    arrow = i.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor('black')
        arrow.set_linewidth(2)
