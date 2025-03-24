import numpy as np
from typing import Tuple

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