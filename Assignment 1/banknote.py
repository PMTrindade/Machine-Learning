import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import accuracy_score

"""

Duvidas:
1 - best_c

"""

"""

1 - PRE-PROCESS
    get_data
    train_test_split - test_size = 0.33; stratification
    n folds = 5 ou 10
    kf = Stratifiedkfold

2 - LOGISTIC REGRESSION
    Build and fine tune Logisitic Regression
        -> find best C param (training and validation sets) -> graph (must return best)
        -> Evaluate -> test set

3 - NEIGHBOORS
    Build and fine tune K-NN
        -> find best k (training and validation sets)
        -> Evaluate -> Test set

4 - NAIVE BAYES
    Build and fine tune Naive Bayes
        -> bandwidth
        -> Evaluate with test set

5 - COMPARISIONS
    Compare the 3 classifiers with Statistical Significance Test

"""

# 1 - PRE-PROCESSING THE DATA
mat = np.loadtxt("/home/bfernandes/School/AA/TP1-data.csv", delimiter=',')

#Suffle / Standardize
data = shuffle(mat)
Ys = data[:,-1]
Xs = data[:,:-1]
means = np.mean(Xs, axis=0)
stdevs = np.std(Xs, axis=0)
Xs = (Xs - means) / stdevs
    
#Split (1/3 test | 2/3 valid)
X_r,X_t,Y_r,Y_t = train_test_split(Xs, Ys, test_size=0.33, stratify = Ys)

   
#Generate folds and loop
folds = 5
kf = StratifiedKFold(Y_r, n_folds=folds)

# 2 - LOGISTIC REGRESSION
best_C = 1e12
best_Err = 1e12
C = 1.0
Cs = []
errs = []
for ix in range(20):
    reg = LogisticRegression(C=C, tol=1e-10)
    scores = cross_val_score(reg, X_r, Y_r, cv = kf)
    va_err = 1 - np.mean(scores)
    #print C,':', va_err
    
    #Find best_c
    if(va_err < best_Err):
        best_C = C
        best_Err = va_err
    
    errs.append(va_err)
    Cs.append(C)
    C = C * 2


fig = plt.figure(figsize = (8,8), frameon = False)
plt.plot(np.log(Cs), errs, '-b', lineWidth = 1)
plt.show()

print 'best_c:', best_C, 'best_Err:', best_Err

reg = LogisticRegression(C=best_C, tol=1e-10)
reg.fit(X_t, Y_t)
test_scores = reg.score(X_t, Y_t)
log_test_err = 1 - np.mean(test_scores)

log_test_predict = reg.predict(X_t)
log_correct = log_test_predict == Y_t
log_wrong = log_test_predict != Y_t


# 3 - K-NN REGRESSION
best_K = 0
best_Err = 1e12
K = 1.0
Ks = []
errs = []
for ix in range(20):
    reg = KNeighborsClassifier(K)
    scores = cross_val_score(reg, X_r, Y_r, cv = kf)
    va_err = 1 - np.mean(scores)
    tr_err = va_err
    #print K,':', va_err
    #Find best_K
    if(va_err < best_Err):
        best_K = K
        best_Err = va_err
    
    errs.append(va_err)
    Ks.append(K)
    K = K + 2
    
fig = plt.figure(figsize = (8,8), frameon = False)
plt.plot(Ks, errs, '-b', lineWidth = 1)
plt.show()

print 'best_K:', best_K, 'best_Err:', best_Err

reg = KNeighborsClassifier(best_K)
reg.fit(X_t, Y_t)
test_scores = reg.score(X_t, Y_t)
knn_test_err = 1 - np.mean(test_scores)

knn_test_predict = reg.predict(X_t)
knn_correct = knn_test_predict == Y_t
knn_wrong = knn_test_predict != Y_t

# 4 - KDENB REGRESSION
class KDENB:

    def __init__(self, bw):
        self.bw = bw
    
    def fit(self, X, Y): 
        #split original Data X by the binary class values
        X0 = X[Y[:]==0, :]
        X1 = X[Y[:]==1, :]
        
        #Calculate priori probability for each class value in log scale
        index0 = np.where(Y_r==0)[0]
        index1 = np.where(Y_r==1)[0]
        self.base0 = np.log((np.shape(index0)[0])/float(np.shape(X_r)[0]))
        self.base1 = np.log((np.shape(index1)[0])/float(np.shape(X_r)[0]))
        self.kdes = []
        
        for ix in range(X0.shape[1]):
            new_kde_tup = []
            kde0 = KernelDensity(kernel = 'gaussian', bandwidth = self.bw)
            kde0.fit(X0[:, [ix]])
            new_kde_tup.append(kde0)
            kde1 = KernelDensity(kernel = 'gaussian', bandwidth = self.bw)
            kde1.fit(X1[:, [ix]])
            new_kde_tup.append(kde1)
            self.kdes.append(new_kde_tup)

    def score(self, X, Y):
        #inic column vector with priori probabilities P0 P1
        p0 = np.ones(X.shape[0]) * self.base0
        p1 = np.ones(X.shape[0]) * self.base1
        for ix in range(X.shape[1]):
            #Evauluate the density model on the framedata to ix
            p0 = p0 + self.kdes[ix][0].score_samples(X[:, [ix]])
            p1 = p1 + self.kdes[ix][1].score_samples(X[:, [ix]])
        
        #Calculate predicted
        cl = np.zeros(len(p0))
        cl[p1 > p0] = 1
        return np.mean(cl == Y)
        
    def predict(self, X):
        p0 = np.ones(X.shape[0]) * self.base0
        p1 = np.ones(X.shape[0]) * self.base1
        for ix in range(X.shape[1]):
            #Evauluate the density model on the framedata to ix
            p0 = p0 + self.kdes[ix][0].score_samples(X[:, [ix]])
            p1 = p1 + self.kdes[ix][1].score_samples(X[:, [ix]])
        
        #Calculate predicted
        cl = np.zeros(len(p0))
        cl[p1 > p0] = 1
        return cl
    
    def get_params(self, deep = True):
        return {"bw": self.bw}
            
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

best_BW = 0
best_Err = 1e12
bw = 0.01
BWs = []
errs = []

while bw < 1:
    reg = KDENB(bw)
    reg.fit(X_r, Y_r)
    scores = cross_val_score(reg, X_r, Y_r, cv = kf)
    va_err = 1 - np.mean(scores)
    tr_err = va_err
    
    #Find best_K
    if(va_err < best_Err):
        best_BW = bw
        best_Err = va_err
    
    errs.append(va_err)
    BWs.append(bw)
    bw = bw + 0.02
    
fig = plt.figure(figsize = (8,8), frameon = False)
plt.plot(BWs, errs, '-b', lineWidth = 1)
plt.show()

reg = KDENB(bw)
reg.fit(X_t, Y_t)
test_scores = reg.score(X_t, Y_t)
nbkde_test_err = 1 - np.mean(test_scores)

nbkde_test_predict = reg.predict(X_t)
nbkde_correct = nbkde_test_predict == Y_t
nbkde_wrong = nbkde_test_predict != Y_t

print 'Logistic Regression Test Error:', log_test_err
print 'K Nearest Neighbors Classifier Test Error:', knn_test_err
print 'Naive Bayes Classifier Test Error:', nbkde_test_err

print "====== McNemar's Test ====="
def mcnemar(e1, e2):
    e01 = e10 = 0
    for ix in range(np.shape(e1)[0]):
        if e1[ix] == True and e2[ix] == False:
            e01 += 1
        if e1[ix] == False and e2[ix] == True:
            e10 += 1
    print e01, e10
    return ((abs(e01 - e10) - 1)** 2) / float(e01 + e10)

print "Logistic Regression VS K Nearest Neighbors Classifier:", mcnemar(log_wrong, knn_wrong)
print "K Nearest Neighbors Classifier VS Naive Bayes Classifier:", mcnemar(knn_wrong, nbkde_wrong)
print "Naive Bayes Classifier VS Logistic Regression:", mcnemar(nbkde_wrong, log_wrong)