import numpy as np
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(f'{filename}')
    return data['features'], data['domains'], data['digits']


features_train, domains_train, digits_train = load_data('train_data.npz')
features_test, domains_test, digits_test = load_data('test_data.npz')


def accuracy(matrix):
    goodSum = 0
    for i in range(10):
        goodSum += matrix[i][i]
    matrix_sum = sum(sum(row) for row in matrix)
    return goodSum / matrix_sum


ratioArray = np.arange(0.1, 0.6, 0.05)
accuracyRes = np.zeros((5, len(ratioArray)))

for i in range(5):
    for j, ratio in enumerate(ratioArray):
        rfc_domain = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42)

        # Fit the classifier to the training data
        rfc_domain.fit(features_train, domains_train)

        def getData(features_train, digits_train, domains_train, domain_idx, ratio):
            all_indices = []
            for k in range(5):
                indices = np.where(domains_train == k)[0]
                if (k != domain_idx):
                    np.random.shuffle(indices)
                    num_samples = int(ratio * len(indices))
                    all_indices = np.append(all_indices, indices[:num_samples])
                else:
                    all_indices = np.append(all_indices, indices)

            all_indices = np.int_(all_indices)

            features_filtered = features_train[all_indices]
            digits_filtered = digits_train[all_indices]
            return features_filtered, digits_filtered

        def dataWithSameDomian(features_train, digits_train, domains_train, domain_idx, ratio=1):
            indices = np.where(domains_train == domain_idx)[0]
            if (ratio != 1):
                num_samples = int(len(indices) * 0.01)  # choose 1% of the data
                indices = np.random.choice(indices, num_samples, replace=False)
            # Filter the data based on the indices
            features_filtered = features_train[indices]
            digits_filtered = digits_train[indices]
            return features_filtered, digits_filtered

        rfc_digits = []

        features_filtered, digits_filtered = getData(features_train, digits_train, domains_train, i, ratio)
        rfc_digit = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42)
        rfc_digit.fit(features_filtered, digits_filtered)
        rfc_digits.append(rfc_digit)
        predicted_domains = rfc_domain.predict(features_test)

        confusion_mat = confusion_matrix(domains_train, predicted_domains)

        features_filtered, digits_filtered = dataWithSameDomian(features_train, digits_train, predicted_domains, i)
        digit_predict = rfc_digit.predict(features_filtered)
        confusion_mat = confusion_matrix(digits_filtered, digit_predict)

        accuracyRes[i][j] = accuracy(confusion_mat)

    plt.plot(ratioArray, accuracyRes[i], label='Group ' + str(i))
    print(i)

plt.legend()
plt.title('Accuracy vs. Ratio for Domain')

plt.xlabel('Ratio')
plt.ylabel('Accuracy')
plt.show()
