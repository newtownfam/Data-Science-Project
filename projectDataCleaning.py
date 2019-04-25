# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:41:58 2019

@author: Ally Nicolella
"""

import pandas as pd
import ast
import numpy as np

original = pd.read_csv('C:/Users/Ally Nicolella/Documents/DS3001/Project/kaggle-datasets/all_kaggle_datasets.csv')

deleted = original.drop(columns = ['creatorName', 'creatorUrl', 'datasetId', 'licenseName', 
                         'maintainerOrganization', 'ownerAvatarUrl', 'ownerUserId', 
                         'ownerUrl', 'scriptsUrl', 'thumbnailImageUrl', 'isPrivate', 
                         'isHidden', 'isFailed', 'isDeleted', 'isCollaborator'])


totalVotes = []
hasAlreadyVotedUp = []
hasAlreadyVotedDown = [] 
canUpvote = []
canDownvote = []
voteUpUrl = []
voteDownUrl = []
voters = []
currentUserInfo = []
showVoters = []
alwaysShowVoters = []

for i in deleted.voteButton:
    dictForm = ast.literal_eval(i)
    totalVotes.append(dictForm['totalVotes'])
    hasAlreadyVotedUp.append(dictForm['hasAlreadyVotedUp'])
    hasAlreadyVotedDown.append(dictForm['hasAlreadyVotedDown'])
    canUpvote.append(dictForm['canUpvote'])
    canDownvote.append(dictForm['canDownvote'])
    voteUpUrl.append(dictForm['voteUpUrl'])
    voteDownUrl.append(dictForm['voteDownUrl'])
    voters.append(dictForm['voters'])
    currentUserInfo.append(dictForm['currentUserInfo'])
    showVoters.append(dictForm['showVoters'])
    alwaysShowVoters.append(dictForm['alwaysShowVoters'])
    
deleted['totalVotes'] = np.array(totalVotes)
deleted['voteUpUrl'] = np.array(voteUpUrl)

fileType = []
fileCount = []
totalSize = []

for j in deleted.commonFileTypes:
    tempFileType = []
    tempFileCount = []
    tempTotalSize = []
    listForm = ast.literal_eval(j)
    for k in listForm:
        tempFileType.append(k['fileType'])
        tempFileCount.append(k['count'])
        tempTotalSize.append(k['totalSize'])
    fileType.append(tempFileType)
    fileCount.append(tempFileCount)
    totalSize.append(tempTotalSize)
    
deleted['fileType'] = np.array(fileType)
deleted['fileCount'] = np.array(fileCount)
deleted['totalSize'] = np.array(totalSize)

categoryName = []
competitionCount = []
for l in deleted.categories:
    tempCategoryName = []
    tempCompCount = []
    listForm2 = ast.literal_eval(l)
    for m in listForm2['categories']:
        tempCategoryName.append(m['name'])
        tempCompCount.append(m['competitionCount'])
    compSum = sum(tempCompCount)
    categoryName.append(tempCategoryName)
    competitionCount.append(compSum)
    
deleted['name'] = np.array(categoryName)
deleted['competitionCount'] = np.array(competitionCount)

cleaned = deleted.drop(columns = ['voteButton', 'commonFileTypes', 'categories'])

cleaned.to_csv('C:/Users/Ally Nicolella/Documents/DS3001/Project/kaggle-datasets/cleanedKaggleData_41719_2.csv')






