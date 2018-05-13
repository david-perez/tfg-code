import argparse
import datetime

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

import logging_utils
from DatabaseManager import DatabaseManager
from SpacyAnalyzer import SpacyAnalyzer


class BagOfWordsGenerator:
    def __init__(self, log, vocabulary, subject_ids, corpus, chart_dates):
        # corpus is a Python list containing medical notes.
        # The note corpus[i] belongs to the patient subject_ids[i] and was taken on chart_dates[i].

        self.__log = log
        self.__vocabulary = vocabulary
        self.__corpus = corpus
        self.__subject_ids = subject_ids
        self.__chart_dates = chart_dates

        self.__analyzer = SpacyAnalyzer()
        self.__X = None

    def build_X(self):
        vectorizer = CountVectorizer(analyzer=self.__analyzer.analyze, vocabulary=self.__vocabulary)
        self.__X = vectorizer.fit_transform(self.__corpus)
        self.__log.info("Term-document count matrix computed")

    def build_bag_of_words_vectors(self):
        if self.__X is None:
            self.build_X()

        ret = []

        rows, cols = self.__X.shape
        subject_id = -1 # Sentinel value.
        bag_of_words = csr_matrix((1, cols))  # Initialised to all zeros.
        how_many_notes = 0
        for i in range(rows):
            if self.__subject_ids[i] != subject_id:
                # Store the bag of words for the previous subject_id.
                if subject_id != -1:
                    bag_of_words_col_ind = bag_of_words.nonzero()[1].tolist()
                    bag_of_words_data = bag_of_words.data.tolist()
                    ret.append((subject_id, how_many_notes, bag_of_words_col_ind, bag_of_words_data))

                # Prepare bag of words for next subject_id.
                subject_id = self.__subject_ids[i]
                bag_of_words = csr_matrix((1, cols))
                how_many_notes = 0
            bag_of_words += self.__X.getrow(i)
            how_many_notes += 1

        # Store the last bag of words of the last patient.
        bag_of_words_col_ind = bag_of_words.nonzero()[1].tolist()
        bag_of_words_data = bag_of_words.data.tolist()
        ret.append((subject_id, how_many_notes, bag_of_words_col_ind, bag_of_words_data))

        return ret

    def build_bag_of_words_vectors_rnn(self):
        if self.__X is None:
            self.build_X()

        ret = []

        rows, cols = self.__X.shape
        for i in range(rows):
            bag_of_words = self.__X.getrow(i)
            subject_id = self.__subject_ids[i]
            chart_date = self.__chart_dates[i]
            bag_of_words_col_ind = bag_of_words.nonzero()[1].tolist()
            bag_of_words_data = bag_of_words.data.tolist()
            ret.append((subject_id, chart_date, bag_of_words_col_ind, bag_of_words_data))

        return ret


def get_table_name(args, experiment_id):
    ret = 'bw_'

    if args.test_set:
        ret += 'test_'
    elif args.validation_set:
        ret += 'val_'
    else:
        ret += 'train_'

    if args.toy_set:
        ret += 'toy_'

    if args.top100_labels:
        ret += 'top10_'
    else:
        ret += 'top100_'

    if args.for_rnn:
        ret += 'rnn_'

    ret += str(experiment_id)

    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate bag of words vectors for each patient using the patient's "
                                                 "medical notes and a provided vocabulary. The bag of words are "
                                                 "stored in a database table.")
    parser.add_argument('vocabulary_experiment', help='the experiment id from the row in the database where the vocabulary to be used is stored')
    parser.add_argument('--toy_set', nargs='?', const=700, help='how many rows to fetch from the corpus table')
    parser.add_argument('--test_set', action='store_true', default=False, help='fetch the notes from the test table')
    parser.add_argument('--validation_set', action='store_true', default=False, help='fetch the notes from the validation table')
    parser.add_argument('--top100_labels', action='store_true', default=False)
    parser.add_argument('--for_rnn', action='store_true', default=False)
    args = parser.parse_args()

    db = DatabaseManager()

    start = datetime.datetime.now()
    time_str = start.strftime("%m%d_%H%M%S")
    config = vars(args)
    experiment_id = db.bag_of_words_generator_experiment_create(config, start)

    log_filename = '{}_bag_of_words_generator.log'.format(experiment_id)
    db.bag_of_words_generator_experiment_insert_log_file(experiment_id, log_filename)

    logger = logging_utils.build_logger(log_filename).getLogger('bag_of_words_generator')
    logger.info('Program start, bag of words generator experiment id = %s', experiment_id)
    logger.info(config)

    vocabulary = db.load_vocabulary(args.vocabulary_experiment)
    logger.info('Vocabulary loaded')
    logger.info('Vocabulary length = %s', len(vocabulary))

    table_name = get_table_name(args, experiment_id)

    # Get the corpus and prepare the bag of words generator.
    db = DatabaseManager()
    subject_ids, corpus, chart_dates = db.get_corpus(toy_set=args.toy_set,
                                                     top100_labels=args.top100_labels,
                                                     validation_set=args.validation_set,
                                                     test_set=args.test_set)
    bag_of_words_generator = BagOfWordsGenerator(logger, vocabulary, subject_ids, corpus, chart_dates)

    if args.for_rnn:
        bag_of_words_vectors_rnn = bag_of_words_generator.build_bag_of_words_vectors_rnn()
        logger.info('Bag of words vectors for RNN created')
        db.insert_bag_of_words_vectors_rnn(bag_of_words_vectors_rnn,
                                           table_name)
        logger.info('Bag of words vectors for RNN inserted in table %s', table_name)
    else:
        bag_of_words_vectors = bag_of_words_generator.build_bag_of_words_vectors()
        logger.info('Bag of words vectors created')
        db.insert_bag_of_words_vectors(bag_of_words_vectors,
                                       table_name)
        logger.info('Bag of words vectors inserted in table %s', table_name)

    end = datetime.datetime.now()
    db.bag_of_words_generator_experiment_insert_table_name(experiment_id, end, table_name)
    logger.info('Bag of words vectors inserted into database, table = %s', table_name)

    print(table_name)