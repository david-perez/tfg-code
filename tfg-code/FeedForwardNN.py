import argparse
import json
from collections import defaultdict

import tensorboardX
import torch
from time import strftime, gmtime

import BagOfWordsVectorsLoader
import logging_utils
from evaluate_classifier import compute_metrics_and_log_to_stdout


def print_cuda_info():
    idx = torch.cuda.current_device()
    logger.info('GPU detected: {}'.format(torch.cuda.get_device_name(idx)))


def determine_tensor_type():
    if not args.no_gpu:
        print_cuda_info()
        logger.info('Running on GPU')
        dtype = torch.cuda.FloatTensor
    else:
        logger.info('Running on CPU')
        dtype = torch.FloatTensor

    return dtype


def load_X_Y(table_name, validation_set=False, test_set=False):
    dtype = determine_tensor_type()

    data, row_ind, col_ind, n_patients, n_features, Y = BagOfWordsVectorsLoader.load_X_Y_nn(table_name,
                                                                                            top100_labels=args.top100_labels,
                                                                                            validation_set=validation_set,
                                                                                            test_set=test_set)
    logger.info('[%s]   Patients: %s, Features: %s', table_name, n_patients, n_features)
    logger.info('[%s]   Bag of words vectors loaded', table_name)
    indices = torch.LongTensor([row_ind, col_ind])
    values = torch.FloatTensor(data)
    X = torch.sparse.FloatTensor(indices, values, torch.Size((n_patients, n_features))).to_dense().type(dtype)  # TODO Cuidado con sparse tensor.
    logger.info('[%s]   X tensor built', table_name)
    Y = torch.FloatTensor(Y).type(dtype)  # BCEWithLogitsLoss requires a FloatTensor.
    logger.info('[%s]   Y tensor built', table_name)

    return torch.autograd.Variable(X), torch.autograd.Variable(Y)


def last_layer_to_predictions(last_layer):
    pred = last_layer.data.sigmoid().numpy().copy()
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    return pred


def log_metrics(metrics, epoch, tb_logger, metric_logger):
    # First, log every single metric separately.
    for metric_name, val in metrics.items():
        tb_logger.add_scalar(metric_name, val, epoch + 1)
        metric_logger[metric_name].append((val, epoch + 1))  # Append metrics from this epoch to previous metrics.

    # Next, log the metrics in useful groups.
    for group in [['precision', 'recall'], ['jaccard', 'subset_accuracy']]:
        tb_logger.add_scalars('-'.join(group), {key: metrics[key] for key in group}, epoch + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feed forward neural network. Reads bag of words vectors from a '
                                                 'training set and a test set, stored in the provided tables, '
                                                 'and evaluates the performance of a fully connected '
                                                 'feed forward neural network.')
    parser.add_argument('train_table_name')
    parser.add_argument('validation_table_name')
    parser.add_argument('test_table_name')
    parser.add_argument('--top100_labels', action='store_true', default=False)
    parser.add_argument('--no_gpu', action='store_true', default=False)
    args = parser.parse_args()

    time = strftime("%m%d_%H%M%S", gmtime())
    root_logger = logging_utils.build_logger('{}_feed_forward.log'.format(time))
    logger = root_logger.getLogger('feed_forward')

    logger.info('Program start')
    logger.info(args)

    logger.info('Building train tensors...')
    X_train, Y_train = load_X_Y(args.train_table_name)
    logger.info('Train tensors built')

    logger.info('Building validation tensors...')
    X_val, Y_val = load_X_Y(args.validation_table_name, validation_set=True)
    logger.info('Validation tensors built')

    N, D_in = X_train.shape  # Number of samples, number of features.
    if args.top100_labels:  # Dimension of the first and second hidden layers, and dimension of the output vector.
        H1, H2, D_out = 1000, 1000, 100
    else:
        H1, H2, D_out = 300, 100, 10

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, H1),
        torch.nn.ReLU(),
        torch.nn.Linear(H1, H2),
        torch.nn.ReLU(),
        torch.nn.Linear(H2, D_out),
    )

    if not args.no_gpu:
        model.cuda()

    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
    learning_rate, decay, momentum = 0.01, 1e-6, 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=decay, momentum=momentum, nesterov=True)

    tb_logger_train = tensorboardX.SummaryWriter(log_dir='../tensorboard_logs/feed_forward_nn_train_' + time)
    tb_logger_val = tensorboardX.SummaryWriter(log_dir='../tensorboard_logs/feed_forward_nn_val_' + time)
    metric_logger_train = defaultdict(list)
    metric_logger_val = defaultdict(list)

    epochs = 200
    for epoch in range(epochs):
        # First of all, train the model using the training set.
        Y_pred_train = model(X_train)
        loss_train = loss_fn(Y_pred_train, Y_train)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        # Now, evaluate the model on the validation set.
        Y_pred_val = model(X_val)
        loss_val = loss_fn(Y_pred_val, Y_val)

        # Finally, log all the data.
        logger.info('Epoch = %s/%s, Loss train = %s, Loss val = %s',
                    epoch + 1, epochs, format(loss_train.data[0], '.5f'), format(loss_val.data[0], '.5f'))

        tb_logger_train.add_scalar('binary_cross_entropy', loss_train.data[0], epoch + 1)
        tb_logger_val.add_scalar('binary_cross_entropy', loss_val.data[0], epoch + 1)

        metrics_train = compute_metrics_and_log_to_stdout(logger, Y_train.data.numpy(), last_layer_to_predictions(Y_pred_train), tag='train')
        metrics_val = compute_metrics_and_log_to_stdout(logger, Y_val.data.numpy(), last_layer_to_predictions(Y_pred_val), tag='val')

        log_metrics(metrics_train, epoch, tb_logger_train, metric_logger_train)
        log_metrics(metrics_val, epoch, tb_logger_val, metric_logger_val)

    metrics_filename = '../metrics/feed_forward_nn_' + time + '.json'
    with open(metrics_filename, 'w') as outfile:
        json.dump(metric_logger_train, outfile)
    logger.info('Model done. Metrics written to %s', metrics_filename)
