from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from typing import Tuple
from sklearn.model_selection import GridSearchCV
# from sklearn.cluster import KMeans

def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(f'{filename}')
    return data['features'], data['domains'], data['digits']

features_train, domains_train, digits_train = \
    load_data('train_data.npz')

print(features_train.shape)
print(np.unique(domains_train, return_counts=True))

features_test, domains_test, digits_test = \
    load_data('test_data.npz')
print(features_test.shape)
print(np.unique(domains_test, return_counts=True))

def accuracy(matrix):
    goodSum = 0
    for i in range(10):
        goodSum += matrix[i][i]
    matrix_sum = sum(sum(row) for row in matrix)
    return (goodSum/matrix_sum)*100

# Train a random forest classifier
# rfc = RandomForestClassifier(n_estimators=100, random_state=42,max_depth=5,min_samples_leaf=20,max_features=10)
# rfc.fit(features_train, digits_train)
# # Use the classifier to predict the labels of the test data
# digits_pred = rfc.predict(features_test)
accuracies =[]
ranges =[]
# for i in range(1,6):
#     rfc = RandomForestClassifier(n_estimators=40, random_state=42,max_depth=i*5,min_samples_leaf=20)
#     rfc.fit(features_train, digits_train)
#     digits_pred = rfc.predict(features_test)
#     cm_initial =confusion_matrix(digits_test,digits_pred)
#     accuracies.append(accuracy(cm_initial))
#     ranges.append(i*5)
# for i in range(1,6):
#     rfc = RandomForestClassifier(n_estimators=30, random_state=42,max_depth=10,min_samples_leaf=20)
#     rfc.fit(features_train, digits_train)
#     digits_pred = rfc.predict(features_test)
#     cm_initial =confusion_matrix(digits_test,digits_pred)
#     accuracies.append(accuracy(cm_initial))
#     ranges.append(i*10)
# for i in range(1,6):
#     rfc = RandomForestClassifier(n_estimators=30, random_state=42,max_depth=10,min_samples_leaf=20,max_features=i*10)
#     rfc.fit(features_train, digits_train)
#     digits_pred = rfc.predict(features_test)
#     cm_initial =confusion_matrix(digits_test,digits_pred)
#     accuracies.append(accuracy(cm_initial))
#     ranges.append(i*10)
for i in range(1,11):
    rfc = RandomForestClassifier(n_estimators=30, random_state=i*10,max_depth=10,min_samples_leaf=20,max_features=10)
    rfc.fit(features_train, digits_train)
    digits_pred = rfc.predict(features_test)
    cm_initial =confusion_matrix(digits_test,digits_pred)
    accuracies.append(accuracy(cm_initial))
    ranges.append(i*10)
# for i in range(1,11):
#     rfc = RandomForestClassifier(n_estimators=i*10, random_state=42,max_depth=5,min_samples_leaf=20)
#     rfc.fit(features_train, digits_train)
#     digits_pred = rfc.predict(features_test)
#     cm_initial =confusion_matrix(digits_test,digits_pred)
#     accuracies.append(accuracy(cm_initial))
plt.plot(ranges,accuracies)
plt.show()
# print(accuracy(cm_initial))
#n_estimators
#max_depth
#min_samples_leaf
#max_features
#random state