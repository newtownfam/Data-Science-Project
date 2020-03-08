# DS3001 Project
## Kaggle Data Project Repository
### Team: Mining for the Minors
- Peter Christakos
- Andrew Morrison
- Ally Nicolella
- Joseph Yuen

### Website Link
https://newtownfam.github.io/Data-Science-Project

### Files
- 1.png - Visual Asset
- 2.png - Visual Asset
- 3.png - Visual Asset
- 4.png - Visual Asset
- background.jpg - Visual Asset
- cleanedKaggleData_41719_version5.csv - cleaned data for creation of decision tree
- cleanedKaggleData_version4.csv - cleaned data for observations on dataset
- Dashboard 1.pdf/png - Tableau observations which may not be accessible on the website due to lack of credentials
- decisionTree.png - predicts whether or not a dataset will be featured or not
- decisionTreeBuilder-noEncoding.ipynb/py - code used to generate decisionTree.png
- decisionTreeBuilder.py - can encode cleanedKaggleData_version4.csv
- index.html - website source code
- kaggleProject.dot - dot file for decisionTree.png
- projectDataCleaning.py - cleans original dataset

### Feature Encoding
#### Topic Count
- 0:0
- [1,5):1
- [5,8):2
- [8,12):3
- [12,45):4
- [45,90):5

#### Data Size
- [0,500):0
- [500,1000):1
- [1000,6000):2
- [6000,16000):3
- [16000,50000):4
- [50000,150000):5
- [150000,300000):6
- [300000,1000000):7
- [1000000,2000000):8
- [2000000,10000000):9
- [10000000,100000000):10
- [100000000,1000000000):11
- [1000000000,10000000000):12
- [10000000000,10000000000):13
- [10000000000,100000000000):14
- [100000000000,1000000000000):15
- [1000000000000,100000000000000):16

#### Competition Count
- 0:0
- [1,3):1
- [3,5):2
- [5,10):3
- [10,45):4
- [45,87):5

#### isFeatured
- False: 0
- True: 1

### Decision Tree Accuracy
87.76%
