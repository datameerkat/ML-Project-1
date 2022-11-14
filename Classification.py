# -*- coding: utf-8 -*-
"""
Script performs classification of obesity level on obesity-dataset with three models: 
    Artificial Neural Network (ANN),
    Multinomial Regression (MR),
    Baseline (BL).
In order to find the best solution double cross validation has been proposed with K1 = K2 = 10 folds. 
Comparision of methods was made with use of McNemar's tests for each pair.
Used data has been standarized and regularized with parameters:
    hidden units for ANN = [1,3,...,19],
    stopping criteria for MR = [10^-5,10^-4,...,10^4].

@author: msalwowski
"""

from ExtractData import * 
from matplotlib.pyplot import figure, show, title
from scipy.io import loadmat
from toolbox_02450 import dbplotf, train_neural_net, visualize_decision_boundary, mcnemar
import numpy as np
import torch
from sklearn import model_selection
import sklearn.linear_model as lm
import time

def ComputeBaselinePredictions(classes):
    counts = np.bincount(classes)
    mode = np.argmax(counts)
    return classes != mode

start = time.time()
y = y_classification

# === PARAMETERS SETTING ===
K1 = 10
K2 = 10
hidden_units = np.arange(1,21,2)
lambdas = np.power(10.,range(-5,5))
n_replicates = 1
max_iter = 10000
# =========

# === PREPARATIONS ===
X = np.concatenate((np.ones((X.shape[0],1)),X),1) # offset feature
M = M+1
CV1 = model_selection.KFold(K1, shuffle=True)
CV2 = model_selection.KFold(K2, shuffle=True)
mu = np.empty((K1, M-1))
sigma = np.empty((K1, M-1))
opt_hidden_units = np.zeros(K1)
opt_lambdas = np.zeros(K1)
ANN_test_errors = np.zeros(K1)
MR_test_errors = np.zeros(K1)
BL_test_errors = np.zeros(K1)
ANN_y_hat = []
MR_y_hat = []
BL_y_hat = []
# =========

i = 0
for train_index, test_index in CV1.split(X,y):
    print("Outer layer [" + str(i) + "]")
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    # === STANDARIZATION ===
    mu[i, :] = np.mean(X_train[:, 1:], 0)
    sigma[i, :] = np.std(X_train[:, 1:], 0)    
    X_train[:, 1:] = (X_train[:, 1:] - mu[i, :] ) / sigma[i, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[i, :] ) / sigma[i, :] 
    # =========
    
    ANN_validation_errors = np.zeros((len(hidden_units), K2))
    MR_validation_errors = np.zeros((len(lambdas), K2))
    
    j = 0
    for inner_train_index, inner_test_index in CV2.split(X_train, y_train):
        print("Inner layer [" + str(j) + "]")
        X_train_inner = X_train[inner_train_index]
        y_train_inner = y_train[inner_train_index]
        X_validation = X_train[inner_test_index]
        y_validation = y_train[inner_test_index]
    
        # === ANN HIDDEN UNITS ===
        for k in range(0, len(hidden_units)):
            model = lambda: torch.nn.Sequential(
                                        torch.nn.Linear(M, hidden_units[k]),
                                        torch.nn.ReLU(), 
                                        torch.nn.Linear(hidden_units[k], C), 
                                        torch.nn.Softmax(dim=1) 
                                        )
            loss_fn = torch.nn.CrossEntropyLoss() # due to multiclass problem
            net, _, _ = train_neural_net(model, loss_fn,
                                          X=torch.tensor(X_train_inner, dtype=torch.float),
                                          y=torch.tensor(y_train_inner, dtype=torch.long),
                                          n_replicates=n_replicates,
                                          max_iter=max_iter)
            softmax_logits = net(torch.tensor(X_validation, dtype=torch.float))
            y_validation_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
            errors_ratio = sum(y_validation_est != y_validation)/len(y_validation)
            ANN_validation_errors[k][j] = errors_ratio # without factor
        # =========
        
        # === MR LAMBDAS ===
        for k in range(0, len(lambdas)):
            logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=lambdas[k], random_state=1, max_iter=max_iter)
            logreg.fit(X_train_inner,y_train_inner)
            errors_ratio = sum(logreg.predict(X_validation)!=y_validation)/len(y_validation)
            MR_validation_errors[k][j] = errors_ratio # without factor
        # =========
        
        j += 1

    # === OPTIMAL ANN MODEL ===
    opt_h = hidden_units[np.argmin(ANN_validation_errors.sum(axis=1))] #select optimal model
    opt_hidden_units[i] = opt_h
    opt_model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, opt_h),
                                torch.nn.ReLU(), 
                                torch.nn.Linear(opt_h, C), 
                                torch.nn.Softmax(dim=1) 
                                )
    loss_fn = torch.nn.CrossEntropyLoss()
    net, _, _ = train_neural_net(opt_model, loss_fn,
                                  X=torch.tensor(X_train, dtype=torch.float),
                                  y=torch.tensor(y_train, dtype=torch.long),
                                  n_replicates=n_replicates,
                                  max_iter=max_iter)
    softmax_logits = net(torch.tensor(X_test, dtype=torch.float))
    y_test_est = (torch.max(softmax_logits, dim=1)[1]).data.numpy()
    ANN_y_hat.extend(y_test_est)
    errors_ratio = sum(y_test_est != y_test)/len(y_test)
    ANN_test_errors[i] = errors_ratio # without factor
    # =========
    
    # === OPTIMAL LAMBDA REGRESSION ===
    opt_l = lambdas[np.argmin(MR_validation_errors.sum(axis=1))]
    opt_lambdas[i] = opt_l
    logreg = lm.LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=opt_l, random_state=1, max_iter=max_iter)
    logreg.fit(X_train,y_train)
    y_test_est = logreg.predict(X_test)
    MR_y_hat.extend(y_test_est)
    errors_ratio = sum(y_test_est!=y_test)/len(y_test)
    MR_test_errors[i] = errors_ratio # without factor
    # =========
    
    # === BASELINE ===
    y_test_est = ComputeBaselinePredictions(y_test)
    BL_y_hat.extend(y_test_est)
    errors_ratio = sum(y_test_est!=y_test)/len(y_test)
    BL_test_errors[i] = errors_ratio
    # =========
    
    i+=1

# === STATISTICAL EVALUATION ===
alpha = 0.05
[ANN_baseline_theta, ANN_baseline_CI, ANN_baseline_p] = mcnemar(y_classification, ANN_y_hat, BL_y_hat, alpha=alpha)
[MR_baseline_theta, MR_baseline_CI, MR_baseline_p] = mcnemar(y_classification, MR_y_hat, BL_y_hat, alpha=alpha)
[ANN_MR_theta, ANN_MR_CI, ANN_MR_p] = mcnemar(y_classification, ANN_y_hat, MR_y_hat, alpha=alpha)
# =========

end = time.time()
elapsed_time = end - start

print()
print("====== RESULTS ======")
print()

print("=== Artificial Neural Network (ANN) ===")
print("ANN errors:")
print(ANN_test_errors)
print("ANN optimal hidden units count:")
print(opt_hidden_units)
print("=========")

print("=== Multinomial Regression (MR) ===")
print("MR errors:")
print(MR_test_errors)
print("MR optimal lambdas:")
print(opt_lambdas)
print("=========")

print("=== Baseline (BL) ===")
print("BL errors:")
print(BL_test_errors)
print("=========")

print()
print("====== MCNEMAR'S TESTS ======")
print()

print("=== ANN vs BL ===")
print("theta: ", ANN_baseline_theta)
print("CI: ", ANN_baseline_CI)
print("p-value: ", ANN_baseline_p)
print("=========")

print("=== MR vs BL ===")
print("theta: ", MR_baseline_theta)
print("CI: ", MR_baseline_CI)
print("p-value: ", MR_baseline_p)
print("=========")

print("=== ANN vs MR ===")
print("theta: ", ANN_MR_theta)
print("CI: ", ANN_MR_CI)
print("p-value: ", ANN_MR_p)
print("=========")

print()

print("=== Time Elapsed ===")
print(str(elapsed_time) + " [s]")
print(str(elapsed_time / 60) + " [m]")
print(str(elapsed_time / 3600) + " [h]")
print("=========")
