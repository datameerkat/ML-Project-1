
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 11:33:18 2022

@author: max44
"""
from ExtractData import * 
from scipy import stats
from matplotlib.pylab import (figure, plot, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import sklearn.linear_model as lm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from toolbox_02450 import rlr_validate, train_neural_net
import torch

##Regression part a: Linear regression with regularization
y = y_regression


# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
#attributeNames = [u'Offset']+attributeNames
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)
#CV = model_selection.KFold(K, shuffle=False)

# Values of lambda
lambdas = np.power(10.,range(-5,5))
#Number of hidden layers - must be length = K!
#hlayer = np.arange(1,21,2)
hlayer = np.arange(1,11,1)
max_iter = 10000
n_replicates = 1

# Initialize variables
Error_train_ANN = np.empty((K,K))
Error_test_ANN = np.empty((K,K))
Error_train_base = np.empty((K,1))
Error_test_base = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))

#Initalize comparison table
comparison = np.empty((K,6))

k=0
for train_index, test_index in CV.split(X,y):
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K)) 
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    

    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)
    
    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)    

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Test error of base model
    Error_train_base[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_base[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    
    #ANN
    l = 0
    for train_index_ANN, test_index_ANN in CV.split(X_train, y_train):   
        # Extract training and test set for current CV fold, convert to tensors
        X_train_ANN = torch.Tensor(X[train_index_ANN,:])
        y_train_ANN = torch.Tensor(np.asmatrix(y[train_index_ANN]).T)
        X_test_ANN = torch.Tensor(X[test_index_ANN,:])
        y_test_ANN = torch.Tensor(np.asmatrix(y[test_index_ANN]).T)
        model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, hlayer[k]), #M features to n_hidden_units
                            torch.nn.ReLU(),   # 1st transfer function,
                            torch.nn.Linear(hlayer[k], hlayer[k]), #1. hidden layer
                            torch.nn.ReLU(),   # 2st transfer function,
                            torch.nn.Linear(hlayer[k], 1), # 2. hidden layer: n_hidden_units to 1 output neuron
                            # no final tranfer function, i.e. "linear output"
                            )
        loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train_ANN,
                                                           y=y_train_ANN,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        # Determine estimated class labels for test set
        y_test_est_ANN = net(X_test_ANN)
        # Determine errors and errors
        se = (y_test_est_ANN.float()-y_test_ANN.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test_ANN)).data.numpy() #mean
        Error_test_ANN[k,l] = mse
#        y_test_error = (y_test_est_ANN.float()-y_test_ANN.float())**2
#        Error_test_ANN[k,l] = (y_test_est_ANN.float()-y_test_ANN.float())**2/test_index_ANN.shape[0] 
        l+=1
    #find best ANN for this outer fold
    opt_nodes = np.argmin(Error_test_ANN[k,:], axis=0)
    
    #update comparison table
    comparison[k-1,0] = k
    comparison[k-1,1] = hlayer[opt_nodes]
    comparison[k-1,2] = Error_test_ANN[k,opt_nodes]
    comparison[k-1,3] = opt_lambda
    comparison[k-1,4] = Error_test_rlr[k]
    comparison[k-1,5] = Error_test_base[k]
    
    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()