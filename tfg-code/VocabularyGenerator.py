import argparse
import datetime

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import logging_utils
from DatabaseManager import DatabaseManager
from SpacyAnalyzer import SpacyAnalyzer


class VocabularyGenerator:
    def __init__(self, corpus, log, tf_idf_cutoff=40000, word_appearance_in_documents_min_threshold=3):
        self.__corpus = corpus
        self.__log = log
        self.__tf_idf_cutoff=tf_idf_cutoff

        self.__analyzer = SpacyAnalyzer()

        # Consider words that appear in at least word_appearance_in_documents_min_threshold documents.
        self.__word_appearance_in_documents_min_threshold = word_appearance_in_documents_min_threshold

    def __build_vocabulary_word_appearance_in_documents(self):
        vectorizer = CountVectorizer(analyzer=self.__analyzer.analyze)
        X = vectorizer.fit_transform(self.__corpus)
        self.__log.info("Term-document count matrix computed")

        appearance_matrix = X > 0
        sum_columns = np.asarray(appearance_matrix.sum(axis=0))[0]
        feature_names = vectorizer.get_feature_names()

        # Build a list of words that appear in at least word_appearance_in_documents_min_threshold documents.
        vocabulary = [feature_names[i] for i, elem in enumerate(sum_columns)
                      if elem >= self.__word_appearance_in_documents_min_threshold]

        return vocabulary

    def __build_vocabulary_tf_idf_scores(self, vocabulary):
        # Recalculate the term-document matrix by TF-IDF, this time using only the built vocabulary.
        vectorizer = TfidfVectorizer(analyzer=self.__analyzer.analyze, vocabulary=vocabulary)
        X = vectorizer.fit_transform(self.__corpus)
        self.__log.info('Term-document TF-IDF matrix computed')

        sum_columns = np.asarray(np.sum(X, axis=0))[0]

        # The vocabulary is selected as the top cutoff terms ordered by TF-IDF scores.
        # We first sort the TF-IDF scores in descending order (reverse=True) and get the indices of the feature_names of
        # the first cutoff terms. With those indices, we look into the feature_names array to obtain the vocabulary.
        feature_names = vectorizer.get_feature_names()
        vocabulary = [feature_names[idx] for idx, _ in
                      sorted(enumerate(sum_columns), key=lambda t: t[1], reverse=True)[:self.__tf_idf_cutoff]]
        self.__log.info('Final vocabulary calculated')

        return vocabulary

    def build_vocabulary(self):
        vocabulary = self.__build_vocabulary_word_appearance_in_documents()
        self.__log.info('There are {} words that appear in at least {} documents.'
              .format(len(vocabulary), self.__word_appearance_in_documents_min_threshold))

        vocabulary = self.__build_vocabulary_tf_idf_scores(vocabulary)
        self.__log.info('The first {} terms (ordered by TF-IDF scores) that appear in at least {} documents of the training set are:'
            .format(min(len(vocabulary), self.__tf_idf_cutoff), self.__word_appearance_in_documents_min_threshold))
        self.__log.info(vocabulary[:100])
        if len(vocabulary) >= 100:
            self.__log.info('(abbreviated, these are only the first 100 terms)')

        return vocabulary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate vocabulary from corpus.')
    parser.add_argument('--toy_set', nargs='?', const=700, help='how many rows to fetch from the corpus table')
    parser.add_argument('--top100_labels', action='store_true', default=False)
    args = parser.parse_args()

    db = DatabaseManager()

    start = datetime.datetime.now()
    time_str = start.strftime("%m%d_%H%M%S")
    config = vars(args)
    experiment_id = db.vocabulary_experiment_create(config, start)

    log_filename = '{}_vocabulary_generator.log'.format(experiment_id)
    db.vocabulary_experiment_insert_log_file(experiment_id, log_filename)

    logger = logging_utils.build_logger(log_filename).getLogger('vocabulary_generator')
    logger.info('Program start, vocabulary experiment id = %s', experiment_id)
    logger.info(config)

    _, corpus, _ = db.get_corpus(toy_set=args.toy_set, top100_labels=args.top100_labels)

    vocabulary_generator = VocabularyGenerator(corpus, logger)
    vocabulary = vocabulary_generator.build_vocabulary()

    end = datetime.datetime.now()
    db.vocabulary_experiment_insert_vocabulary(experiment_id, end, vocabulary)
    logger.info('Vocabulary inserted into database')
