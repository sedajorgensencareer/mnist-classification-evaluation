from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from s02_load_mnist import create_mnist_test_train


X_train, X_test, y_train, y_test = create_mnist_test_train()

svc_clf = SVC(random_state=42)
svc_clf.fit(X_train[:2000], y_train[:2000])

some_digit = X_train[0]

print(svc_clf.predict([some_digit]))
some_digit_scores = svc_clf.decision_function([some_digit])
print(some_digit_scores.round(2))

class_id = some_digit_scores.argmax()
print(class_id)

print(svc_clf.classes_)
print(svc_clf.classes_[class_id])


## Output all 45 OvO scores
# svc_clf.decision_function_shape = "ovo"
# some_digit_scores_ovo = svc_clf.decision_function([some_digit])
# print(some_digit_scores_ovo.round(2))

from sklearn.multiclass import OneVsRestClassifier

over_clf = OneVsRestClassifier(SVC(random_state=42))
over_clf.fit(X_train[:2000], y_train[:2000])
print(over_clf.predict([some_digit]))
len(over_clf.estimators_)


sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([some_digit]))
print(sgd_clf.decision_function([some_digit]).round())

cv_score = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print(cv_score)



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
scaled_cv_score = cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
print(scaled_cv_score)