import sklearn


def log_metrics(logger, Y_true, Y_pred):
    """Calculate and log metrics evaluating the performance of the results from a multiclass multilabel classifier.

    :param Y_true: array-like, shape = (n_samples, n_classes)
    :param Y_pred: array-like, shape = (n_samples, n_classes)
    """

    precision, recall, fscore, _ = sklearn.metrics.precision_recall_fscore_support(Y_true, Y_pred, average='samples')
    subset_accuracy = sklearn.metrics.accuracy_score(Y_true, Y_pred)
    jaccard = sklearn.metrics.jaccard_similarity_score(Y_true, Y_pred)
    logger.info('F-score = %s, Precision = %s, Recall = %s, Subset-accuracy = %s, Jaccard index = %s',
                fscore, precision, recall, subset_accuracy, jaccard)
