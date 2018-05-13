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


def last_layer_to_predictions(last_layer):
    """Convert outputs from the last layer of the neural network (real numbers) to binary values using a sigmoid function.

    :param last_layer:
    :return:
    """
    pred = last_layer.data.sigmoid().numpy().copy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    return pred
