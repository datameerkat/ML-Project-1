# Project 1

import numpy as np
import pandas as pd

# Load the Project csv data using the Pandas library
filename = 'C:/Users/max44/anaconda3/02450Toolbox_Python/Data/ProjectData.csv'
df = pd.read_csv(filename)

# Regression or Classification
task = "Regression"

# Convert to numpy array
raw_data = df.values  

# Create X
cols = range(0, len(df.columns)) 
X = raw_data[:, cols]


# Extract attribute names of header
attributeNames = np.asarray(df.columns[cols])

## Transform string features into numbers

# family_history_with_overweight
column = 4
classLabels = raw_data[:,column]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
X[:,column] = np.array([classDict[cl] for cl in classLabels])

# FAVC
column = 5
classLabels = raw_data[:,column]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
X[:,column] = np.array([classDict[cl] for cl in classLabels])

# CAEC
column = 8
classLabels = raw_data[:,column]
classDict = dict({'no': 0 ,'Sometimes': 1,'Frequently': 2, 'Always': 3})
X[:,column] = np.array([classDict[cl] for cl in classLabels])

# SMOKE
column = 9
classLabels = raw_data[:,column]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
X[:,column] = np.array([classDict[cl] for cl in classLabels])

# SCC
column = 11
classLabels = raw_data[:,column]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
X[:,column] = np.array([classDict[cl] for cl in classLabels])

# CALC
column = 14
classLabels = raw_data[:,column]
classDict = dict({'no': 0 ,'Sometimes': 1,'Frequently': 2, 'Always': 3})
X[:,column] = np.array([classDict[cl] for cl in classLabels])

#Gender: One-Out-Off K Coding: First transform string into int, than K coding
column = 0
classLabels = raw_data[:,column]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
Gender_num = np.array([classDict[cl] for cl in classLabels])
# One out off K coding
K = Gender_num.max()+1
Gender_encoding = np.zeros((Gender_num.size, K))
Gender_encoding[np.arange(Gender_num.size), Gender_num] = 1
X = np.concatenate( (Gender_encoding, X[:,1:]), axis=1)
#update attributesNames
attributeNames = np.concatenate((["Female","Male"],attributeNames[1:]), axis=0)

#MTRANS: One-Out-Off K Coding: First transform string into int, than K coding
column = 15
classLabels = raw_data[:,column]
classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
MTRANS_num = np.array([classDict[cl] for cl in classLabels])
# One out off K coding
K = MTRANS_num.max()+1
MTRANS_encoding = np.zeros((MTRANS_num.size, K))
MTRANS_encoding[np.arange(MTRANS_num.size), MTRANS_num] = 1
X = np.concatenate( (X[:, :-2], MTRANS_encoding), axis=1)
#update attributesNames
attributeNames = np.concatenate((attributeNames[0:-2],["Automobile","Bike","Motorbike","Public_Transportation","Walking"]), axis=0)

#targetName_r = attributeNames_c[2]
#attributeNames_r = np.concatenate((attributeNames_c[[0, 1, 3]], classNames), axis=0)


# get Y data and delete Y data from X
y_regression = raw_data[:,3]
y_classification = raw_data[:,16]
X = np.delete(X, [4], 1)
# delete attributeNames of Y-columns
attributeNames = np.delete(attributeNames, [4], 0)











