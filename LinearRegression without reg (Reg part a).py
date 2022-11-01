# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 11:33:18 2022

@author: max44
"""
from ExtractData import * 
from scipy import stats
from matplotlib.pyplot import figure, plot, xlabel, ylabel, legend, show
import sklearn.linear_model as lm
import numpy as np
from sklearn.preprocessing import StandardScaler

##Regression part a: Linear regression
y = y_regression

#Standardize data
Xscaler = StandardScaler()
X = Xscaler.fit_transform(X)
#Yscaler = StandardScaler()
#y = Yscaler.fit_transform(y.reshape(-1,1))

#LinearRegression model
model = lm.LinearRegression(fit_intercept=True)
model = model.fit(X,y)
# Compute model output:
y_est = model.predict(X)

#Rescale data
#X = Xscaler.inverse_transform(X)
#y = Yscaler.inverse_transform(y)
#y_est = Yscaler.inverse_transform(y_est)

#Get weights of attributes
weights  = model.coef_

# Plot true vs estimated data
f = figure()
plot(y,y_est,'.')
plot(np.arange(20,170,1),np.arange(20,170,1),'-')
xlabel('true weight'); ylabel('estimated weight')
#legend(['Training data', 'Regression fit (model)'])

show()