# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:10:00 2019

@author: Ally Nicolella

Production of a decision tree model to be used to help users predict,
based on past Kaggle datasets, if their dataset will become "Super Featured" on
the Kaggle website.
"""

import pandas as pd
import numpy as np
from sklearn.externals.six import StringIO
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
import pydotplus
from IPython.display import Image

col_names = ['datasetSize','topicCount','competitionCount','isFeatured']
cleaned = pd.read_csv("cleanedKaggleData_41719_version5.csv", header=None, names=col_names)

'''
Encode topic count with following cutoffs/values:
    (0):0
    (1,4):1
    (5,7):2
    (8,11):3
    (12,44):4
    (45,90):5
'''

'''
Encode dataSize with following cutoffs/values:
    [0,500):0
    [500,1000):1
    [1000,6000):2
    [6000,16000):3
    [16000,50000):4
    [50000,150000):5
    [150000,300000):6
    [300000,1000000):7
    [1000000,2000000):8
    [2000000,10000000):9
    [10000000,100000000):10
    [100000000,1000000000):11
    [1000000000,10000000000):12
    [10000000000,10000000000):13
    [10000000000,100000000000):14
    [100000000000,1000000000000):15
    [1000000000000,100000000000000):16
'''

'''
Encoded competitionCount using following cutoffs/values:
    [0]:0
    [1,3):1
    [3,5):2
    [5,10):3
    [10,45):4
    [45,87):5
'''

'''
Encoded isFeatured using following cutoffs/values:
    False: 0
    True: 1
'''

'''
Split the dataset into a training and test set in order to build and test a
decision tree model for prediction. Print accuracy of the model.
'''

msk = np.random.rand(len(cleaned)) < 0.8

training_set = cleaned[msk]
test_set = cleaned[~msk]

features = ['datasetSize', 'topicCount','competitionCount']
y = training_set['isFeatured']
X = training_set[features]
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X,y)
y_pred = dt.predict(test_set[features])
y_test = test_set['isFeatured']
print(metrics.accuracy_score(y_test, y_pred))

with open("kaggleProject.dot", 'w') as f:
    export_graphviz(dt, out_file=f, feature_names=features)

# export png
dot_data = StringIO()
tree.export_graphviz(dt, out_file=dot_data,
                filled=True, rounded=False, feature_names = features,class_names=['isNotFeatured','Featured'],impurity=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('decisionTree.png')
Image(graph.create_png())
