import numpy as np
import pickle

from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from DatabaseManager import DatabaseManager

from timing import log
from utils import get_icd9_codes_map


def load_X_train_Y_train(toy_set=None, top10_labels=True, top100_labels=False):
    assert(top10_labels ^ top100_labels)

    db = DatabaseManager()
    cur = db.get_bag_of_words_vectors_training_set(toy_set=toy_set, top10_labels=top10_labels)

    # Y_train holds the Y vector for each of the 10 (100) binary classifiers.
    Y_train = [np.zeros(cur.rowcount) for _ in range(10 if top10_labels else 100)]

    icd9_codes_map = get_icd9_codes_map(top10_labels=top10_labels, top100_labels=top100_labels)

    data = []
    row_ind = []
    col_ind = []
    cnt = 0
    for subject_id, _, bag_of_words_binary_vector_col_ind, bag_of_words_binary_vector_data in cur:
        bag_of_words_vector_col_ind = pickle.loads(bag_of_words_binary_vector_col_ind)
        bag_of_words_vector_data = pickle.loads(bag_of_words_binary_vector_data)

        data += bag_of_words_vector_data
        row_ind += [cnt] * len(bag_of_words_vector_col_ind)
        col_ind += bag_of_words_vector_col_ind

        # Get the icd9 codes of the diseases this subject_id has.
        diagnoses = db.get_icd9_codes(subject_id=subject_id)
        for icd9_code in diagnoses:
            idx = icd9_codes_map[icd9_code]
            Y_train[idx][cnt] = 1

        cnt += 1

    return csr_matrix((data, (row_ind, col_ind))), Y_train


if __name__ == '__main__':
    X_train, Y_train = load_X_train_Y_train(toy_set=100, top10_labels=True)
    log('X_train, Y_train loaded')

    classifiers = []
    for Y in Y_train:
        logistic_regression_model = LogisticRegression()
        logistic_regression_model.fit(X_train, Y)
        classifiers.append(logistic_regression_model)
