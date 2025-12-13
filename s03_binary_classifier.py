from matplotlib import pyplot as plt
import numpy as np
from s01_config import save_fig
from s02_load_mnist import create_mnist_test_train

X_train, X_test, y_train, y_test = create_mnist_test_train()

y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)


some_digit = X_train[0]
# print(sgd_clf.predict([some_digit]))

from sklearn.model_selection import cross_val_score
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# from sklearn.dummy import DummyClassifier

# dummy_clf = DummyClassifier(strategy="most_frequent", random_state=42)
# dummy_clf.fit(X_train, y_train_5)

# print(cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy"))

# from sklearn.model_selection import StratifiedKFold
# from sklearn.base import clone

# skfolds = StratifiedKFold(n_splits=3)  
# for train_index, test_index in skfolds.split(X_train, y_train_5):
#     clone_clf = clone(sgd_clf)
#     X_train_folds = X_train[train_index]
#     y_train_folds = y_train_5[train_index]
#     X_test_fold = X_train[test_index]
#     y_test_fold = y_train_5[test_index]

#     clone_clf.fit(X_train_folds, y_train_folds)
#     y_pred = clone_clf.predict(X_test_fold)
#     n_correct = sum(y_pred == y_test_fold)
#     print(n_correct / len(y_pred))



from sklearn.model_selection import cross_val_predict

y_train_pred =cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)


from sklearn.metrics import confusion_matrix


# Creates a confusion matrix representing the performance of predictions
# Top left is true negatives, top right false positives
# Bottom left is false negatives, bottom right is true positives
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)


from sklearn.metrics import precision_score, recall_score

print(precision_score(y_train_5, y_train_pred))

## Manual prediction calculation from matrix
# precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])

print(recall_score(y_train_5, y_train_pred))


from sklearn.metrics import f1_score
print(f1_score(y_train_5, y_train_pred))


## Shows decision score of an individual instance indicating how strongly a sample belongs
## to a positive or negative class, and that the prediction is made by comparing this score to a chosen threshold.
# y_scores = sgd_clf.decision_function([some_digit])
# print(y_scores)

# threshold = 0
# y_some_digit_pred = (y_scores > threshold)

# print(y_some_digit_pred)


from sklearn.metrics import precision_recall_curve
threshold = 0


y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

## Plot precision against threshold & recall against threshold
# plt.figure(figsize=(8, 4))  # Create plotting canvas
# plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2) # Plot precision vs threshold tradeoff
# plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2) # Plot recall vs threshold tradeoff
# plt.vlines(threshold, 0, 1.0, "k", "dotted", label="threshold") # Certain threshold value inspection

# idx = (thresholds >= threshold).argmax()  # Finds the index of the first threshold >= current threshold
# plt.plot(thresholds[idx], precisions[idx], "bo") # Mark precision at chosen threshold
# plt.plot(thresholds[idx], recalls[idx], "go") # Mark recall at chosen threshold

# # Final formatting: sets axis, adds grids and lines, and saves figure
# plt.axis([-50000, 50000, 0, 1]) 
# plt.grid()
# plt.xlabel("Threshold")
# plt.legend(loc="center right")
# save_fig("precision_recall_vs_threshold_plot")

# plt.show()


## Plot Precision / Recall Curve
# plt.plot(recalls, precisions, linewidth=2, label="Precision/Recall Curve")
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.axis([0, 1, 0, 1])
# plt.legend(loc="lower left")
# plt.grid()

# save_fig("Precision_vs_Recall_Plot")
# plt.show()



## Plot Precision / Recall Curve with Colour visualisation reflecting threshold along curve
# clip = 10000
# thr = np.clip(thresholds, -clip, clip)

# plt.figure(figsize=(7, 5))
# sc = plt.scatter(recalls[:-1], precisions[:-1],
#                  c=thr, cmap="viridis", s=10)

# plt.colorbar(sc, label=f"Decision threshold (clipped to ±{clip})")
# plt.xlabel("Recall"); plt.ylabel("Precision")
# plt.axis([0,1,0,1]); plt.grid()
# save_fig("Precision_vs_Recall_with_Threshold")
# # plt.show()


## Find the lowest threshold that gives 90% precision
idx_for_90_prediction = (precisions >= 0.90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_prediction]
print(threshold_for_90_precision)

y_train_pred_90 = (y_scores >= threshold_for_90_precision)

print("Precision:", precision_score(y_train_5, y_train_pred_90))
print("Recall:", recall_score(y_train_5, y_train_pred_90))