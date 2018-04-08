from time import strftime, gmtime

import numpy as np
import pickle

from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

import logging_utils
from DatabaseManager import DatabaseManager

from utils import get_icd9_codes_map


def load_X_Y(table_name, top10_labels=True, top100_labels=False, test_set=False, n_features=None):
    """Return the sample matrix X and a list of Y vectors for the bag of words vectors contained in :param table_name.

    The bag of words vectors are loaded from :param table_name and returned in X, a sparse matrix where each row is
    a bag of words vector, associated to a unique patient. The columns of the matrix are the occurrences of each
    particular word of the vocabulary in each bag of words vector. The bag of words vectors are not normalized.

    The list of Y vectors is returned as a list of NumPy arrays. Each Y vector is a binary vector of length equal to
    the number of rows (number of bag of words vectors) in X, indicating whether a patient has that particular label
    (ICD9 code) or not. The list has thus length 10 or 100, the number of classifiers trained (one for each label),
    depending on whether :param top10_labels or :param top100_labels is True, respectively.

    :param table_name: the name of the table where the bag of words vectors are stored.
    :param top10_labels:
    :param top100_labels:
    :param test_set:
    :param n_features: If provided and not None, the returned X matrix will have :param n_features columns. Otherwise,
    the matrix will have as many columns as necessary. For example, if the training set of bag of words vectors and the
    test set of bag of words vectors were generated using the same vocabulary, the X matrix for both sets should have
    the same number of columns. However, if no bag of words vectors from the test set have occurrences of the last words
    in the vocabulary, the X matrix associated with the test set will have less columns than the X matrix associated
    with the training set. Providing a value for :param n_features will eliminate this problem.
    :return: X, Y
    """
    assert(top10_labels ^ top100_labels)

    db = DatabaseManager()
    cur = db.get_bag_of_words_vectors(table_name)

    # Y_list holds the Y vector for each of the 10 (100) binary classifiers (list of numpy arrays).
    Y_list = [np.zeros(cur.rowcount) for _ in range(10 if top10_labels else 100)]

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
        diagnoses = db.get_icd9_codes(subject_id=subject_id, test_set=test_set)
        for icd9_code in diagnoses:
            idx = icd9_codes_map[icd9_code]
            Y_list[idx][cnt] = 1

        cnt += 1

    if n_features is not None:
        return csr_matrix((data, (row_ind, col_ind)), shape=(cnt, n_features)), Y_list
    else:
        return csr_matrix((data, (row_ind, col_ind))), Y_list


def calculate_metrics(predicted, actual):
    """Calculate metrics evaluating the performance of the logistic regression classifiers.

    :param predicted: array-like, shape = (n_samples, n_classifiers)
    :param actual: array-like, shape = (n_samples, n_classifiers)
    :return: precision, recall
    """

    precision = 0
    recall = 0
    number_of_patients = predicted.shape[0]
    for i in range(number_of_patients):
        # nonzero(a) returns a tuple of arrays, one for each dimension of a, containing the indices of the non-zero elements in that dimension.
        # That is why we select the 0th element of the result to the call to nonzero.
        predicted_labels = predicted[i, :].nonzero()[0]
        actual_labels = actual[i, :].nonzero()[0]
        intersection = np.intersect1d(predicted_labels, actual_labels, assume_unique=True)

        if len(predicted_labels) > 0:
            precision += len(intersection) / len(predicted_labels)

        if len(actual_labels) > 0:
            recall += len(intersection) / len(actual_labels)

    precision /= number_of_patients
    recall /= number_of_patients

    return precision, recall


if __name__ == '__main__':
    time = strftime("%Y%m%d%H%M%S", gmtime())
    root_logger = logging_utils.build_logger('logistic_regression_{}.log'.format(str(time)))
    logger = root_logger.getLogger('logistic_regression')
    top10_labels = True
    top100_labels = False
    bag_of_words_train_table_name = 'bw_vocabulary_train_top10_labels_20180405161930p'
    bag_of_words_test_table_name = 'bw_test_vocabulary_train_top10_labels_20180405161930p'
    # bag_of_words_train_table_name = 'bw_vocabulary_train_toy_top10_labels_20180406113245p'
    # bag_of_words_test_table_name = 'bw_test_vocabulary_train_toy_top10_labels_20180406113245p'

    logger.info('Program start')
    logger.info('Config: top10_labels = %s, top100_labels = %s, bag_of_words_train_table_name = %s, bag_of_words_test_table_name = %s' ,
                top10_labels, top100_labels, bag_of_words_train_table_name, bag_of_words_test_table_name)
    X_train, Y_train = load_X_Y(bag_of_words_train_table_name, top10_labels=top10_labels, top100_labels=top100_labels)
    n_features = X_train.shape[1]
    logger.info('X_train, Y_train loaded')
    normalize(X_train, norm='l1', axis=1, copy=False)
    logger.info('X_train normalized')

    classifiers = []
    for i, Y in enumerate(Y_train):
        logistic_regression_model = LogisticRegression()
        logistic_regression_model.fit(X_train, Y)
        logger.info('Classifier {} trained'.format(i))
        classifiers.append(logistic_regression_model)

    logger.info('Building result matrix for training set')
    number_of_patients_train = Y_train[0].shape[0]
    predicted_matrix_train = np.zeros((number_of_patients_train, len(classifiers)))
    actual_matrix_train = np.zeros((number_of_patients_train, len(classifiers)))

    for j, classifier, actual_Y in zip(range(len(classifiers)), classifiers, Y_train):
        predicted = classifier.predict(X_train)  # An array containing which patients are tagged with the label of this classifier.
        predicted_matrix_train[:, j] = predicted
        actual_matrix_train[:, j] = actual_Y

    precision_train, recall_train = calculate_metrics(predicted_matrix_train, actual_matrix_train)
    logger.info('Computing metrics for training set')
    logger.info('Training set -- Precision = %s, Recall = %s', precision_train, recall_train)

    X_test, Y_test = load_X_Y(bag_of_words_test_table_name, top10_labels=top10_labels, top100_labels=top100_labels,
                              test_set=True, n_features=n_features)
    logger.info('X_test, Y_test loaded')
    normalize(X_test, norm='l1', axis=1, copy=False)
    logger.info('X_test normalized')

    logger.info('Building result matrix for test set')
    number_of_patients_test = Y_test[0].shape[0]
    predicted_matrix_test = np.zeros((number_of_patients_test, len(classifiers)))
    actual_matrix_test = np.zeros((number_of_patients_test, len(classifiers)))

    for j, classifier, actual_Y in zip(range(len(classifiers)), classifiers, Y_test):
        predicted = classifier.predict(X_test)  # An array containing which patients are tagged with the label of this classifier.
        predicted_matrix_test[:, j] = predicted
        actual_matrix_test[:, j] = actual_Y

    precision_test, recall_test = calculate_metrics(predicted_matrix_test, actual_matrix_test)
    logger.info('Computing metrics for test set')
    logger.info('Test set -- Precision = %s, Recall = %s', precision_test, recall_test)
