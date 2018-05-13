import argparse
import datetime
from collections import defaultdict

import numpy as np
import tensorboardX
import torch

import logging_utils
import tensor_loader
from database_manager import DatabaseManager
from rnn_model import RNNModel
from evaluate_classifier import compute_metrics_and_log_to_stdout, last_layer_to_predictions


def log_metrics(metrics, chunk, epoch, metric_logger, tb_logger=None):
    # First, log every single metric separately.
    for metric_name, val in metrics.items():
        if tb_logger is not None:
            tb_logger.add_scalar(metric_name, val, chunk * epochs + epoch + 1)
        metric_logger[metric_name].append((val, chunk * epochs + epoch + 1))  # Append metrics from this epoch to previous metrics.

    if tb_logger is None:
        return

    # Next, log the metrics in useful groups.
    for group in [['precision', 'recall'], ['jaccard', 'subset_accuracy']]:
        tb_logger.add_scalars('-'.join(group), {key: metrics[key] for key in group}, chunk * epochs + epoch + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recurrent neural network. Reads bag of words vectors from a '
                                                 'training set, validation set and a test set, stored in the provided tables, '
                                                 'and evaluates the performance of a recursive neural network.')
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
    experiment_id = db.classifier_experiment_create(config, start, 'rnn', args.train_table_name, args.val_table_name, args.test_table_name)

    log_filename = '{}_rnn.log'.format(experiment_id)
    db.classifier_experiment_insert_log_file(experiment_id, log_filename)

    logger = logging_utils.build_logger(log_filename).getLogger('rnn')
    logger.info('Program start, classifier experiment id = %s', experiment_id)
    logger.info(args)

    # We can't fit all of the notes into memory. Split the patients into chunks.
    # Ensure 1 < (number of patients / total_chunks).
    total_chunks = 3  # TODO move to program args.

    # Load the first chunk to get number of input features.
    X_train, Y_train = tensor_loader.load_X_Y_rnn(logger, args.train_table_name, chunk=0, total_chunks=total_chunks, no_gpu=args.no_gpu)
    X_val, Y_val = tensor_loader.load_X_Y_rnn(logger, args.val_table_name, chunk=0, total_chunks=total_chunks, no_gpu=args.no_gpu, validation_set=True)

    N, seq_length, D_in = X_train.shape  # Number of samples, sequence length, number of features.
    if args.top100_labels:  # Dimension of the hidden units, and dimension of the output vector.
        H, D_out = 1000, 100
    else:
        H, D_out = 100, 10

    model = RNNModel(D_in, H, D_out)

    if not args.no_gpu:
        model.cuda()

    loss_fn = torch.nn.BCEWithLogitsLoss(size_average=True)
    learning_rate, decay, momentum = 0.01, 1e-6, 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=decay, momentum=momentum, nesterov=True)

    tb_logger_train = tensorboardX.SummaryWriter(log_dir='../tensorboard_logs/rnn_train_' + str(experiment_id))
    tb_logger_val = tensorboardX.SummaryWriter(log_dir='../tensorboard_logs/rnn_val_' + str(experiment_id))
    metrics_train = defaultdict(list)
    metrics_val = defaultdict(list)
    metrics_test = defaultdict(list)

    epochs = 3  # TODO move to program args
    for chunk in range(total_chunks):
        if chunk > 0:  # Load next chunk (first chunk will already be loaded).
            X_train, Y_train = tensor_loader.load_X_Y_rnn(logger, args.train_table_name, chunk=chunk, total_chunks=total_chunks, no_gpu=args.no_gpu)
            X_val, Y_val = tensor_loader.load_X_Y_rnn(logger, args.val_table_name, chunk=chunk, total_chunks=total_chunks, no_gpu=args.no_gpu, validation_set=True)

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
            logger.info('Chunk= %s/%s, Epoch = %s/%s, Loss train = %s, Loss val = %s',
                        chunk + 1, total_chunks, epoch + 1, epochs,
                        format(loss_train.data[0], '.5f'), format(loss_val.data[0], '.5f'))

            tb_logger_train.add_scalar('binary_cross_entropy', loss_train.data[0], chunk * epochs + epoch + 1)
            tb_logger_val.add_scalar('binary_cross_entropy', loss_val.data[0], chunk * epochs + epoch + 1)

            metrics_train_this_epoch = compute_metrics_and_log_to_stdout(logger, Y_train.data.numpy(), last_layer_to_predictions(Y_pred_train), tag='train')
            metrics_val_this_epoch = compute_metrics_and_log_to_stdout(logger, Y_val.data.numpy(), last_layer_to_predictions(Y_pred_val), tag='val')

            log_metrics(metrics_train_this_epoch, chunk, epoch, metrics_train, tb_logger=tb_logger_train)
            log_metrics(metrics_val_this_epoch, chunk, epoch, metrics_val, tb_logger=tb_logger_val)

    # Training done. Evaluate classifier using test set.
    loss_fn_no_average = torch.nn.BCEWithLogitsLoss(size_average=False)
    loss_test = 0
    Y_test_list = []
    Y_pred_test_list = []
    for chunk in range(total_chunks):
        X_test_chunk, Y_test_chunk = tensor_loader.load_X_Y_rnn(logger, args.test_table_name, chunk=chunk, total_chunks=total_chunks, no_gpu=args.no_gpu, test_set=True)
        Y_pred_test_chunk = model(X_test_chunk)
        Y_test_list.append(Y_test_chunk.data.cpu().numpy())
        Y_pred_test_list.append(Y_pred_test_chunk.data.cpu().numpy())

        loss_test += loss_fn_no_average(Y_pred_test_chunk, Y_test_chunk)

        logger.info('Chunk = %s/%s, Accumulated loss test = %s', chunk + 1, total_chunks, format(loss_test.data[0], '.5f'))

    Y_test = np.vstack(tuple(Y_test_list))
    Y_pred_test = np.vstack(tuple(Y_pred_test_list))

    log_metrics(compute_metrics_and_log_to_stdout(logger,
                                                     Y_test,
                                                     last_layer_to_predictions(torch.autograd.Variable(torch.from_numpy(Y_pred_test))),
                                                     tag='test'),
                0,
                0,
                metrics_test)

    end = datetime.datetime.now()
    db.classifier_experiment_insert_metrics(experiment_id, metrics_train, metrics_val, metrics_test, end)
    logger.info('Model done. Metrics written to database')
