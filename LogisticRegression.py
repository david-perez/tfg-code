import argparse
from time import strftime, gmtime

import numpy as np
import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

import logging_utils
from BagOfWordsVectorsLoader import load_X_Y
from evaluate_classifier import log_metrics


def train_classifiers(X_train, Y_train):
    classifiers = []
    for i, Y in enumerate(Y_train):
        logistic_regression_model = LogisticRegression()
        logistic_regression_model.fit(X_train, Y)
        logger.info('Classifier {} trained'.format(i + 1))
        classifiers.append(logistic_regression_model)

    return classifiers


def calculate_predicted_matrix(classifiers, number_of_patients, X_train):
    predicted_matrix = np.zeros((number_of_patients, len(classifiers)))

    for j, classifier in enumerate(classifiers):
        predicted = classifier.predict(X_train)  # An array containing which patients are tagged with the label of this classifier.
        predicted_matrix[:, j] = predicted

    return predicted_matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic regression model. Reads bag of words vectors from a '
                                                 'training set and a test set, stored in the provided tables, '
                                                 'and evaluates the performance of a collection of logistic '
                                                 'regression classifiers, one for each ICD9 label.')
    parser.add_argument('train_table_name')
    parser.add_argument('test_table_name')
    parser.add_argument('--top100_labels', action='store_true', default=False)
    parser.add_argument('--dont_normalize_input_features', action='store_true', default=False)
    args = parser.parse_args()

    time = strftime("%m%d_%H%M%S", gmtime())
    root_logger = logging_utils.build_logger('{}_logistic_regression.log'.format(time))
    logger = root_logger.getLogger('logistic_regression')

    logger.info('Program start')
    logger.info(args)

    X_train, Y_train = load_X_Y(args.train_table_name, top100_labels=args.top100_labels)
    n_features = X_train.shape[1] # TODO This is correct but it would be nicer if we knew the vocabulary length beforehand and provided it to load_X_Y()
    logger.info('X_train, Y_train loaded')
    if not args.dont_normalize_input_features:
        normalize(X_train, norm='l1', axis=1, copy=False)
        logger.info('X_train normalized')

    classifiers = train_classifiers(X_train, Y_train)

    logger.info('Building result matrix for training set')
    number_of_patients_training_set = Y_train[0].shape[0]
    predicted_matrix_train = calculate_predicted_matrix(classifiers, number_of_patients_training_set, X_train)
    actual_matrix_train = np.column_stack(Y_train)  # Stack vertically the elements of the Y_train list.

    logger.info('Computing metrics for training set')
    log_metrics(logger, actual_matrix_train, predicted_matrix_train)

    X_test, Y_test = load_X_Y(args.test_table_name, top100_labels=args.top100_labels, test_set=True, n_features=n_features)
    logger.info('X_test, Y_test loaded')
    if not args.dont_normalize_input_features:
        normalize(X_test, norm='l1', axis=1, copy=False)
        logger.info('X_test normalized')

    logger.info('Building result matrix for test set')
    number_of_patients_test_set = Y_test[0].shape[0]
    predicted_matrix_test = calculate_predicted_matrix(classifiers, number_of_patients_test_set, X_test)
    actual_matrix_test = np.column_stack(Y_test)

    logger.info('Computing metrics for test set')
    log_metrics(logger, actual_matrix_test, predicted_matrix_test)
