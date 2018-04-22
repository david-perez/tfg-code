import sklearn


def log_metrics(logger, Y_true, Y_pred):
    """Calculate and log metrics evaluating the performance of the results from a multiclass multilabel classifier.

    :param Y_true: array-like, shape = (n_samples, n_classes)
    :param Y_pred: array-like, shape = (n_samples, n_classes)
    """

    precision, recall, f1score, _ = sklearn.metrics.precision_recall_fscore_support(Y_true, Y_pred, average='samples')
    subset_accuracy = sklearn.metrics.accuracy_score(Y_true, Y_pred)
    jaccard = sklearn.metrics.jaccard_similarity_score(Y_true, Y_pred)
    logger.info('F1-score = %s, Precision = %s, Recall = %s, Subset-accuracy = %s, Jaccard index = %s',
                f1score, precision, recall, subset_accuracy, jaccard)

    return {'f1score': f1score, 'precision': precision, 'recall': recall, 'subset_accuracy': subset_accuracy, 'jaccard': jaccard}
