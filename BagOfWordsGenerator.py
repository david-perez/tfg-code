import argparse
import json
import os
from time import gmtime, strftime

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
    with open(filename, 'r') as infile:
        return json.load(infile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate bag of words vectors for each patient using the patient's "
                                                 "medical notes and a provided vocabulary. The bag of words are "
                                                 "stored in a database table.")
    parser.add_argument('vocabulary_filename', help='the name of the file where the vocabulary to be used is stored')
    parser.add_argument('--toy_set', nargs='?', const=700, help='how many rows to fetch from the corpus table')
    parser.add_argument('--test_set', action='store_true', default=False, help='fetch the notes from the test table')
    parser.add_argument('--top100_labels', action='store_true', default=False)
    args = parser.parse_args()

    time = strftime("%m%d_%H%M%S", gmtime())
    root_logger = logging_utils.build_logger('{}_bag_of_words_generator.log'.format(time))
    logger = root_logger.getLogger('bag_of_words_generator')

    logger.info('Program start')
    logger.info(args)

    vocabulary = load_vocabulary(args.vocabulary_filename)
    logger.info('Vocabulary loaded')
    logger.info('Vocabulary length = %s', len(vocabulary))

    # Get the filename without extension from the path.
    vocabulary_filename = os.path.splitext(os.path.basename(args.vocabulary_filename))[0]

    db = DatabaseManager()
    subject_ids, corpus = db.get_corpus(toy_set=args.toy_set, top100_labels=args.top100_labels, test_set=args.test_set)

    bag_of_words_generator = BagOfWordsGenerator(logger, vocabulary, subject_ids, corpus)
    bag_of_words_vectors = bag_of_words_generator.build_bag_of_words_vectors()
    logger.info('Bag of words vectors created')

    table_name = db.insert_bag_of_words_vectors(bag_of_words_vectors, vocabulary_filename, test_set=args.test_set)
    logger.info('Bag of words vectors inserted in table %s', table_name)
    print(table_name)