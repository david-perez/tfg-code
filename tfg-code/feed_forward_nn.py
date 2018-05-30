import argparse
import datetime
from collections import defaultdict

import tensorboardX
import torch

import logging_utils
import tensor_loader
from database_manager import DatabaseManager
from evaluate_classifier import compute_metrics_and_log_to_stdout, last_layer_to_predictions


def log_metrics(metrics, epoch, metric_logger, tb_logger=None):
    # First, log every single metric separately.
    for metric_name, val in metrics.items():
        if tb_logger is not None:
            tb_logger.add_scalar(metric_name, val, epoch + 1)
        metric_logger[metric_name].append((val, epoch + 1))  # Append metrics from this epoch to previous metrics.

    if tb_logger is None:
        return

    # Next, log the metrics in useful groups.
    for group in [['precision', 'recall'], ['jaccard', 'subset_accuracy']]:
        tb_logger.add_scalars('-'.join(group), {key: metrics[key] for key in group}, epoch + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feed forward neural network. Reads bag of words vectors from a '
                                                 'training set, validation set and a test set, stored in the provided tables, '
                                                 'and evaluates the performance of a fully connected '
                                                 'feed forward neural network.')
    parser.add_argument('train_table_name')
    parser.add_argument('val_table_name')
    parser.add_argument('test_table_name')
    parser.add_argument('--top100_labels', action='store_true', default=False)
    parser.add_argument('--no_gpu', action='store_true', default=False)
    args = parser.parse_args()

    db = DatabaseManager()

    start = datetime.datetime.now()
    time_str = start.strftime("%m%d_%H%M%S")
    config = vars(args)
    experiment_id = db.classifier_experiment_create(config, start, 'nnff', args.train_table_name, args.val_table_name, args.test_table_name)

    log_filename = '{}_nnff.log'.format(experiment_id)
    db.classifier_experiment_insert_log_file(experiment_id, log_filename)

    logger = logging_utils.build_logger(log_filename).getLogger('feed_forward')
    logger.info('Program start, classifier experiment id = %s', experiment_id)
    logger.info(args)

    X_train, Y_train = tensor_loader.load_X_Y(logger, args.train_table_name, args.no_gpu)
    X_val, Y_val = tensor_loader.load_X_Y(logger, args.val_table_name, args.no_gpu, validation_set=True)

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

    tb_logger_train = tensorboardX.SummaryWriter(log_dir='../tensorboard_logs/nnff_train_' + str(experiment_id))
    tb_logger_val = tensorboardX.SummaryWriter(log_dir='../tensorboard_logs/nnff_val_' + str(experiment_id))
    metrics_train = defaultdict(list)
    metrics_val = defaultdict(list)
    metrics_test = defaultdict(list)

    epochs = 1  # TODO move to args.
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

        metrics_train_this_epoch = compute_metrics_and_log_to_stdout(logger, Y_train.cpu().data.numpy(), last_layer_to_predictions(Y_pred_train), tag='train')
        metrics_val_this_epoch = compute_metrics_and_log_to_stdout(logger, Y_val.cpu().data.numpy(), last_layer_to_predictions(Y_pred_val), tag='val')

        log_metrics(metrics_train_this_epoch, epoch, metrics_train, tb_logger_train)
        log_metrics(metrics_val_this_epoch, epoch, metrics_val, tb_logger_val)

    # Training done. Evaluate classifier using test set.
    X_test, Y_test = tensor_loader.load_X_Y(logger, args.test_table_name, args.no_gpu, test_set=True)
    Y_pred_test = model(X_test)
    loss_test = loss_fn(Y_pred_test, Y_test)
    logger.info('Loss test = %s', format(loss_test.data[0], '.5f'))
    log_metrics(compute_metrics_and_log_to_stdout(logger,
                                                  Y_test.cpu().data.numpy(),
                                                  last_layer_to_predictions(Y_pred_test),
                                                  tag='test'),
                0,
                metrics_test)

    end = datetime.datetime.now()
    db.classifier_experiment_insert_metrics(experiment_id, metrics_train, metrics_val, metrics_test, end)
    logger.info('Model done. Metrics written to database')
