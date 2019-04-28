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
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import sklearn

cleaned = pd.read_csv('cleanedKaggleData_41719_version4.csv')

'''
Encode topic count with following cutoffs/values:   
    (0):0
    (1,4):1
    (5,7):2
    (8,11):3
    (12,44):4
    (45,90):5 
'''

encoded_topicCountList = []

for k in cleaned['topicCount']:
    if k == 0:
        encoded_topicCountList.append('0')
    elif k in range(1,5):
        encoded_topicCountList.append('1')
    elif k in range(5,8):
        encoded_topicCountList.append('2')
    elif k in range(8,12):
        encoded_topicCountList.append('3')
    elif k in range(12,45):
        encoded_topicCountList.append('4')
    elif k in range(45,90):
        encoded_topicCountList.append('5')
    else:
        break

cleaned['Encoded topicCount'] = encoded_topicCountList


# Size range: 22 - 33140193553554
'''
Encode dataSize with following cutoffs/values:
    (0,500):0
    (500,1000):1
    (1000,6000):2
    (6000,16000):3
    (16000,50000):4
    (50000,150000):5
    (150000,300000):6
    (300000,1000000):7
    (1000000,2000000):8
    (2000000,10000000):9
    (10000000,100000000):10
    (100000000,1000000000):11
    (1000000000,10000000000):12
    (10000000000,10000000000):13
    (10000000000,100000000000):14
    (100000000000,1000000000000):15
    (1000000000000,100000000000000):16
'''

encoded_datasetSizeList = []

for i in cleaned['datasetSize']:
    if i in range(0,500):
        encoded_datasetSizeList.append('0')
    elif i in range(500,1000):
        encoded_datasetSizeList.append('1')
    elif i in range(1000,6000):
        encoded_datasetSizeList.append('2')
    elif i in range(6000,16000):
        encoded_datasetSizeList.append('3')
    elif i in range(16000,50000):
        encoded_datasetSizeList.append('4')
    elif i in range(50000,150000):
        encoded_datasetSizeList.append('5')
    elif i in range(150000,300000):
        encoded_datasetSizeList.append('6')
    elif i in range(300000,1000000):
        encoded_datasetSizeList.append('7')
    elif i in range(1000000,2000000):
        encoded_datasetSizeList.append('8')
    elif i in range(2000000,10000000):
        encoded_datasetSizeList.append('9')
    elif i in range(10000000,100000000):
        encoded_datasetSizeList.append('10')
    elif i in range(100000000,1000000000):
        encoded_datasetSizeList.append('11')
    elif i in range(1000000000,10000000000):
        encoded_datasetSizeList.append('12')
    elif i in range(10000000000,10000000000):
        encoded_datasetSizeList.append('13')
    elif i in range(10000000000,100000000000):
        encoded_datasetSizeList.append('14')
    elif i in range(100000000000,1000000000000):
        encoded_datasetSizeList.append('15')
    elif i in range(1000000000000,100000000000000):
        encoded_datasetSizeList.append('16')
    else:
        break

cleaned['Encoded datasetSize'] = encoded_datasetSizeList

'''
Encoded competitionCount using following cutoffs/values:
    (0):0
    (1,2):1
    (3,4):2
    (5,9):3
    (10,45):4
    (46,87):5  
'''

encoded_competitionCountList = []

for l in cleaned['competitionCount']:
    if l == 0:
        encoded_competitionCountList.append('0')
    elif l in range(1,3):
        encoded_competitionCountList.append('1')
    elif l in range(3,5):
        encoded_competitionCountList.append('2')
    elif l in range(5,10):
        encoded_competitionCountList.append('3')
    elif l in range(10,45):
        encoded_competitionCountList.append('4')
    elif l in range(45,115):
        encoded_competitionCountList.append('5')
    else:
        break

cleaned['Encoded competitionCount'] = encoded_competitionCountList

encoded_isFeatured = {
        True : '0',
        False : '1'
        }

cleaned['Encoded isFeatured'] = cleaned['isFeatured'].map(encoded_isFeatured)

'''
Split the dataset into a training and test set in order to build and test a 
decision tree model for prediction. Print accuracy of the model.
'''

msk = np.random.rand(len(cleaned)) < 0.8

training_set = cleaned[msk]
test_set = cleaned[~msk]

features = ['Encoded topicCount','Encoded datasetSize', 'Encoded competitionCount']
y = training_set['Encoded isFeatured']
X = training_set[features]
dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
dt.fit(X,y)
y_pred = dt.predict(test_set[features])
y_test = test_set['Encoded isFeatured']
print(sklearn.metrics.accuracy_score(y_test, y_pred))

with open("kaggleProject.dot", 'w') as f:
    export_graphviz(dt, out_file=f, feature_names=features)
