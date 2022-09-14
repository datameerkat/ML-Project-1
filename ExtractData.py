# Project 1
# Here we extract our data out of the CSV to get X and y (for regression
# and classification, respectively) and also N, M and C.
#%%

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
#%% Transform string features into numbers

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
attributeNames = np.concatenate((attributeNames[0:-2],["Automobile","Bike","Motorbike",\
                                                       "Public_Transportation","Walking"]), axis=0)

#%% get Y data, transform Y data for classification and finalize attributesName
# get Y data

y_regression = raw_data[:,3]
y_classification = raw_data[:,16]

# One-Ouf-Off K coding for y_classification (NObeyesdad)
column = 16
classLabels = raw_data[:,column]
classNames = np.unique(classLabels)
classDict = dict({'Insufficient_Weight': 0 ,'Normal_Weight': 1,'Overweight_Level_I': 2, \
                  'Overweight_Level_II': 3, 'Obesity_Type_I': 4, 'Obesity_Type_II': 5, \
                      'Obesity_Type_III': 6})
Obesity_num = np.array([classDict[cl] for cl in classLabels])
# One out off K coding
K = Obesity_num.max()+1
Obesity_encoding = np.zeros((Obesity_num.size, K))
Obesity_encoding[np.arange(Obesity_num.size), Obesity_num] = 1
y_classification = Obesity_encoding
attributesNames_classification = np.array(['Insufficient_Weight','Normal_Weight','Overweight_Level_I', \
                  'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', \
                      'Obesity_Type_III'])

#delete weight in X and attributeNames (NObeyesdad already got deleted)
X = np.delete(X, [4], 1)
attributeNames = np.delete(attributeNames, [4], 0)

#get number of observations N and number of features M
N, M = X.shape

#get number of classes for classification
C = len(attributesNames_classification)











