import pickle
from datetime import datetime

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

import logging_utils
from DatabaseManager import DatabaseManager
from SpacyAnalyzer import SpacyAnalyzer


class BagOfWordsGenerator:
    def __init__(self, log, vocabulary, subject_ids, corpus):
        # corpus is a Python list containing medical notes.
        # The note corpus[i] belongs to the patient subject_ids[i].

        self.__log = log
        self.__vocabulary = vocabulary
        self.__corpus = corpus
        self.__subject_ids = subject_ids

        self.__analyzer = SpacyAnalyzer()

    def build_bag_of_words_vectors(self):
        vectorizer = CountVectorizer(analyzer=self.__analyzer.analyze, vocabulary=self.__vocabulary)
        X = vectorizer.fit_transform(self.__corpus)
        self.__log.info("Term-document count matrix computed")

        ret = []

        rows, cols = X.shape
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
            bag_of_words += X.getrow(i)
            how_many_notes += 1

        # Store the last bag of words of the last patient.
        bag_of_words_col_ind = bag_of_words.nonzero()[1].tolist()
        bag_of_words_data = bag_of_words.data.tolist()
        ret.append((subject_id, how_many_notes, bag_of_words_col_ind, bag_of_words_data))

        self.__log.info("Bag of words vectors created")

        return ret


def load_vocabulary(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


if __name__ == '__main__':
    root_logger = logging_utils.build_logger('bag_of_words_generator_{}.log'.format(str(datetime.now())))
    logger = root_logger.getLogger('bag_of_words_generator')
    test_set = True
    toy_set = 5000 #  700
    top10_labels = True
    top100_labels = False
    vocabulary_filename = 'vocabulary_train_toy_top10_labels_20180406113245.p'

    logger.info('Program start')
    logger.info('Config: toy_set = %s, test_set = %s, top10_labels = %s, top100_labels = %s, vocabulary_filename = %s',
                toy_set, test_set, top10_labels, top100_labels, vocabulary_filename)

    vocabulary = load_vocabulary('serialized_vocabularies/' + vocabulary_filename)
    logger.info('Vocabulary loaded')
    logger.info('Vocabulary length = %s', len(vocabulary))

    db = DatabaseManager()
    subject_ids, corpus = db.get_corpus(toy_set=toy_set, top10_labels=top10_labels, top100_labels=top100_labels, test_set=test_set)

    bag_of_words_generator = BagOfWordsGenerator(logger, vocabulary, subject_ids, corpus)
    bag_of_words_vectors = bag_of_words_generator.build_bag_of_words_vectors()
    logger.info('Bag of words vectors created')

    table_name = db.insert_bag_of_words_vectors(bag_of_words_vectors, vocabulary_filename, test_set=test_set)
    logger.info('Bag of words vectors inserted in table %s', table_name)
