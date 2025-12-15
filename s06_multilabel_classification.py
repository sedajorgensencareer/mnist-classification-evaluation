import numpy as np
from sklearn.calibration import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from s02_load_mnist import create_mnist_test_train

X_train, X_test, y_train, y_test = create_mnist_test_train() # Create train and test sets

y_train_large = (y_train >= '7') # Create binary labels for digits (large or not)
y_train_odd = (y_train.astype('int8') % 2 == 1) # Create binary labels for digits (odd or even)
y_multilabel = np.c_[y_train_large, y_train_odd] # Combine into multilabel target

## Create and train KNeighboursClassifier multi-label model
knn_clf = KNeighborsClassifier() 
knn_clf.fit(X_train, y_multilabel)



some_digit = X_train[0]
# print(knn_clf.predict([some_digit]))


## Use cross-validation to create predictions across to techniques (macro and weighted)
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1s = f1_score(y_multilabel, y_train_knn_pred, average="macro")
f1sw = f1_score(y_multilabel, y_train_knn_pred, average="weighted")
print(f1s)
print(f1sw)


## Use a classifier chain to provide the model information of the previous label prediction
from sklearn.multioutput import ClassifierChain

chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])

print(chain_clf.predict([some_digit]))
