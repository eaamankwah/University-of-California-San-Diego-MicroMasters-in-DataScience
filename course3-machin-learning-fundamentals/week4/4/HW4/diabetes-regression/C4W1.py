# -*- coding: utf-8 -*-
"""
Created on Sun Dec 4 22:05:54 2016

@author: EAmankwah
"""

# -*- coding: utf-8 -*-

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

# bug fix for display formats to avoid run time errors
pd.set_option('display.float_format', lambda x:'%.2f'%x)

#os.chdir("C:\Decission TREES")

"""
Data Engineering and Analysis
"""
#Load the dataset

ad = pd.read_csv('esearch_data.csv')

data_clean = ad.dropna()

data_clean.dtypes
data_clean.describe()

# categories response variable into binary variable
def internetgrp (row):
   if row['internetuserate'] <= 35.633:
      return 0
   else:
      return 1

"""
Modeling and Prediction
"""
#Split into training and testing sets

predictors = data_clean[['incomeperperson','alcconsumption','armedforcesrate'
'breastcancerper100th','co2emissions','femaleemployrate','hivrate','lifeexpectancy',
'oilperperson','polityscore','relectricperperson','suicideper100th','employrate',
'urbanrate']]

targets = data_clean.internetgrp

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.4)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
classifier=DecisionTreeClassifier()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)

#Displaying the decision tree
from sklearn import tree
#from StringIO import StringIO
from io import StringIO
#from StringIO import StringIO
from IPython.display import Image
out = StringIO()
tree.export_graphviz(classifier, out_file=out)
import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
Image(graph.create_png())
