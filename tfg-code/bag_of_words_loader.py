import pickle

import numpy as np
from scipy.sparse import csr_matrix

from database_manager import DatabaseManager
from utils import get_icd9_codes_map


def load_X_Y(table_name, top100_labels=False, validation_set=False, test_set=False, n_features=None):
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
    db = DatabaseManager()
    cur = db.get_bag_of_words_vectors(table_name)

    # Y_list holds the Y vector for each of the 10 (100) binary classifiers (list of numpy arrays).
    Y_list = [np.zeros(cur.rowcount) for _ in range(100 if top100_labels else 10)]

    icd9_codes_map = get_icd9_codes_map(top100_labels=top100_labels)

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
        diagnoses = db.get_icd9_codes(subject_id=subject_id, validation_set=validation_set, test_set=test_set)
        for icd9_code in diagnoses:
            idx = icd9_codes_map[icd9_code]
            Y_list[idx][cnt] = 1

        cnt += 1

    # TODO assert(cnt == cur.rowcount)

    if n_features is not None:
        return csr_matrix((data, (row_ind, col_ind)), shape=(cnt, n_features)), Y_list
    else:
        return csr_matrix((data, (row_ind, col_ind))), Y_list


def load_X_Y_nn(table_name, top100_labels=False, validation_set=False, test_set=False):
    db = DatabaseManager()
    cur = db.get_bag_of_words_vectors(table_name)
    n_patients = cur.rowcount

    Y = np.zeros((n_patients, 100 if top100_labels else 10))

    icd9_codes_map = get_icd9_codes_map(top100_labels=top100_labels)

    data = []
    row_ind = []
    col_ind = []
    for cnt, (subject_id, _, bag_of_words_binary_vector_col_ind, bag_of_words_binary_vector_data) in enumerate(cur):
        bag_of_words_vector_col_ind = pickle.loads(bag_of_words_binary_vector_col_ind)
        bag_of_words_vector_data = pickle.loads(bag_of_words_binary_vector_data)

        data += bag_of_words_vector_data
        row_ind += [cnt] * len(bag_of_words_vector_col_ind)
        col_ind += bag_of_words_vector_col_ind

        # Get the icd9 codes of the diseases this subject_id has.
        diagnoses = db.get_icd9_codes(subject_id=subject_id, validation_set=validation_set, test_set=test_set)
        for icd9_code in diagnoses:
            idx = icd9_codes_map[icd9_code]
            Y[cnt][idx] = 1

    return data, row_ind, col_ind, n_patients, 40000, Y  # TODO


def load_X_Y_rnn(table_name, chunk_idx, total_chunks, top100_labels=False, validation_set=False, test_set=False):
    db = DatabaseManager()

    subject_ids = db.unique_subject_ids(table_name)
    chunked = np.array_split(np.array(subject_ids), total_chunks)
    subject_id_chunk = chunked[chunk_idx]
    m_subject_id_to_idx = dict()
    for i, subject_id in enumerate(subject_id_chunk):
        m_subject_id_to_idx[subject_id] = i

    n_patients = subject_id_chunk.shape[0]

    Y = np.zeros((n_patients, 100 if top100_labels else 10))

    icd9_codes_map = get_icd9_codes_map(top100_labels=top100_labels)

    first_patient, last_patient = subject_id_chunk[0].item(), subject_id_chunk[-1].item()

    seq_length = 20  # TODO
    n_features = 40000  # TODO
    ret = np.zeros((n_patients, seq_length, n_features))

    cur = db.get_bag_of_words_vectors_rnn(table_name, first_patient, last_patient)

    for note_in_seq, row_id, subject_id, chart_date, bag_of_words_binary_vector_col_ind, bag_of_words_binary_vector_data in cur:
        bag_of_words_vector_col_ind = pickle.loads(bag_of_words_binary_vector_col_ind)
        bag_of_words_vector_data = pickle.loads(bag_of_words_binary_vector_data)

        for col_ind, data in zip(bag_of_words_vector_col_ind, bag_of_words_vector_data):
            ret[m_subject_id_to_idx[subject_id]][note_in_seq - 1][col_ind] = data

        # Get the icd9 codes of the diseases this subject_id has.
        diagnoses = db.get_icd9_codes(subject_id=subject_id, validation_set=validation_set, test_set=test_set)
        for icd9_code in diagnoses:
            idx = icd9_codes_map[icd9_code]
            Y[m_subject_id_to_idx[subject_id]][idx] = 1

    return ret, n_patients, n_features, Y
