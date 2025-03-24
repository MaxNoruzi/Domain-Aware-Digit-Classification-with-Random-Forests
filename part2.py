import numpy as np
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


############################################


def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(f'{filename}')
    return data['features'], data['domains'], data['digits']


features_train, domains_train, digits_train = \
    load_data('train_data.npz')

features_test, domains_test, digits_test = \
    load_data('test_data.npz')


#################################################################

def accuracy(matrix):
    goodSum = 0
    for i in range(10):
        goodSum += matrix[i][i]
    matrix_sum = sum(sum(row) for row in matrix)
    return goodSum / matrix_sum


#####################################################################
print('start')
rfc_domain = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42,max_features=500)

# Fit the classifier to the training data
rfc_domain.fit(features_train, domains_train)

print('afterDomain')


###############################
def dataWithSameDomian(features_train, digits_train, domains_train, domain_idx):
    indices = np.where(domains_train == domain_idx)[0]
    # Filter the data based on the indices
    features_filtered = features_train[indices]
    digits_filtered = digits_train[indices]
    return features_filtered, digits_filtered


rfc_digits = []

for i in range(5):
    features_filtered, digits_filtered = dataWithSameDomian(features_train, digits_train, domains_train, i)
    rfc_digit = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42)
    rfc_digit.fit(features_filtered, digits_filtered)
    rfc_digits.append(rfc_digit)
print('predict start')
predicted_domains = rfc_domain.predict(features_test)

print('domains')
confusion_mat = confusion_matrix(domains_train, predicted_domains)
print(confusion_mat)

for i in range(5):
    rfc = rfc_digits[i]
    features_filtered, digits_filtered = dataWithSameDomian(features_train, digits_train, predicted_domains, i)
    digit_predict = rfc.predict(features_filtered)
    confusion_mat = confusion_matrix(digits_filtered, digit_predict)
    print(confusion_mat)
    print(accuracy(confusion_mat))
