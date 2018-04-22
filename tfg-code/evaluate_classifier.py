import sklearn.metrics


def compute_metrics_and_log_to_stdout(logger, Y_true, Y_pred, tag='train'):
    """Calculate and log metrics evaluating the performance of the results from a multiclass multilabel classifier.

    :param Y_true: array-like, shape = (n_samples, n_classes)
    :param Y_pred: array-like, shape = (n_samples, n_classes)
    """

    precision, recall, f1score, _ = sklearn.metrics.precision_recall_fscore_support(Y_true, Y_pred, average='samples')
    subset_accuracy = sklearn.metrics.accuracy_score(Y_true, Y_pred)
    jaccard = sklearn.metrics.jaccard_similarity_score(Y_true, Y_pred)
    logger.info('[%s]  F1-score = %s, Precision = %s, Recall = %s, Subset-accuracy = %s, Jaccard index = %s',
                tag,
                format(f1score, '.5f'),
                format(precision, '.5f'),
                format(recall, '.5f'),
                format(subset_accuracy, '.5f'),
                format(jaccard, '.5f'))

    return {'f1score': f1score, 'precision': precision, 'recall': recall, 'subset_accuracy': subset_accuracy, 'jaccard': jaccard}
