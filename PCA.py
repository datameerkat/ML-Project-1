"""
In order to obtain correct plots comment out the classification section in ExtractData.py
"""

from ExtractData import *
import scipy.linalg as linalg
import matplotlib.pyplot as plt

# convertion of array of objects to array of floats
Xf = X.astype(float)
N,M = Xf.shape

# =============================================================================
# =============================================================================
# !CAUTION!
# plots each pair of the attributes
# better leave commented
# =============================================================================
# =============================================================================

# for i in range(0,M):
#     for j in range(0,M):
#         plt.plot(Xf[:, i], Xf[:, j], 'o', alpha=0.1)
#         plt.xlabel(attributeNames[i])
#         plt.ylabel(attributeNames[j])
#         plt.show()

# =============================================================================
# =============================================================================
# variance explained by PCA
# =============================================================================
# =============================================================================

# normalization
Xf = Xf - np.ones((N,1))*Xf.mean(axis=0)
Xf = Xf*(1/np.std(Xf,0))

U,S,V = linalg.svd(Xf,full_matrices=False)

rho = (S*S) / (S*S).sum() 

threshold = 0.9

plt.figure()
plt.plot(range(1,len(rho)+1),rho,'o-')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual'])
plt.grid()
plt.show()

plt.figure()
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Cumulative','Threshold'])
plt.grid()
plt.show()

# =============================================================================
# =============================================================================
# the ammount of variation explained as the number of PCA components included
# =============================================================================
# =============================================================================

pcs = [[0,1,2],[3,4,5]]
for pcs_triple in pcs:
    legendStrs = ['PC'+str(e+1) for e in pcs_triple]
    c = ['r','g','b']
    bw = .2
    r = np.arange(1,M+1)
    for i in pcs_triple:    
        plt.bar(r+i*bw, V[:,i], width=bw)
        plt.xticks(r+bw, r)
        # with attributes name (its unreadable):
        # plt.xticks(r+bw, attributeNames[:])
    plt.xlabel('Attribute index')
    plt.ylabel('Component coefficients')
    plt.legend(legendStrs)
    plt.grid()
    plt.title('PCA Component Coefficients')
    plt.show()

# =============================================================================
# =============================================================================
# plot of PCA's against each other
# =============================================================================
# =============================================================================

V = V.T
Z = Xf @ V

pairs = [[0,1],[1,2],[0,2]]
for pair in pairs:
    f = plt.figure()
    plt.title('PCA')
    # for c in range(0,C-1,2):
    for c in range(0,C):
        print(c)
        # class1_mask = y_classification[:,c]==1
        # class2_mask = y_classification[:,c+1]==1
        # class_mask = class1_mask | class2_mask
        class_mask = y_classification[:,c]==1
        plt.plot(Z[class_mask,pair[0]], Z[class_mask,pair[1]], 'o', alpha=.5)
    plt.xlabel('PC{0}'.format(pair[0]+1))
    plt.ylabel('PC{0}'.format(pair[1]+1))
    plt.legend(['Normal', 'Mid', "High"])
    plt.legend(classNames)
    plt.show()

