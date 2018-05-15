import torch

import bag_of_words_loader


def print_cuda_info(logger):
    idx = torch.cuda.current_device()
    logger.info('GPU detected: {}'.format(torch.cuda.get_device_name(idx)))


def determine_tensor_type(logger, no_gpu):
    if not no_gpu:
        print_cuda_info(logger)
        logger.info('Running on GPU')
        dtype = torch.cuda.FloatTensor
    else:
        logger.info('Running on CPU')
        dtype = torch.FloatTensor

    return dtype


def load_X_Y_rnn(logger, table_name, chunk, total_chunks, no_gpu=False, top100_labels=False, validation_set=False, test_set=False):
    tensor_purpose = 'train'
    if validation_set:
        tensor_purpose = 'validation'
    elif test_set:
        tensor_purpose = 'test'

    logger.info('Building %s tensors...', tensor_purpose)

    dtype = determine_tensor_type(logger, no_gpu)

    data, n_patients, n_features, Y = bag_of_words_loader.load_X_Y_rnn(table_name,
                                                                       chunk,
                                                                       total_chunks=total_chunks,
                                                                       top100_labels=top100_labels,
                                                                       validation_set=validation_set,
                                                                       test_set=test_set)
    logger.info('[%s]   Patients: %s, Features: %s', table_name, n_patients, n_features)
    logger.info('[%s]   Bag of words vectors loaded', table_name)
    X = torch.FloatTensor(data)
    logger.info('[%s]   X tensor built', table_name)
    Y = torch.FloatTensor(Y).type(dtype)  # BCEWithLogitsLoss requires a FloatTensor.
    logger.info('[%s]   Y tensor built', table_name)

    logger.info('%s tensors built', tensor_purpose)

    return torch.autograd.Variable(X), torch.autograd.Variable(Y)


def load_X_Y(logger, table_name, no_gpu=False, top100_labels=False, validation_set=False, test_set=False):
    tensor_purpose = 'train'
    if validation_set:
        tensor_purpose = 'validation'
    elif test_set:
        tensor_purpose = 'test'

    logger.info('Building %s tensors...', tensor_purpose)

    dtype = determine_tensor_type(logger, no_gpu)

    data, row_ind, col_ind, n_patients, n_features, Y = bag_of_words_loader.load_X_Y_nn(table_name,
                                                                                        top100_labels=top100_labels,
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

    logger.info('%s tensors built', tensor_purpose)

    return torch.autograd.Variable(X), torch.autograd.Variable(Y)