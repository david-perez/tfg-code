from datetime import datetime
from time import strftime, gmtime

import numpy as np
import pickle
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
            .format(self.__tf_idf_cutoff, self.__word_appearance_in_documents_min_threshold))
        self.__log.info(vocabulary)

        return vocabulary


def write_vocabulary_to_file(vocabulary, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(vocabulary, fp)


def get_filename_vocabulary(time, toy_set=None, top10_labels=True, top100_labels=False):
    assert(top10_labels ^ top100_labels)

    filename = 'serialized_vocabularies/vocabulary_train'
    if toy_set:
        filename += '_toy'
    filename += '_top10_labels_' if top10_labels else '_top100_labels_'
    filename += time
    filename += '.p'

    return filename


if __name__ == '__main__':
    time = strftime("%Y%m%d%H%M%S", gmtime())
    root_logger = logging_utils.build_logger('{}_vocabulary_generator.log'.format(time))
    logger = root_logger.getLogger('vocabulary_generator')
    toy_set = 700
    top10_labels = True
    top100_labels = False

    logger.info('Program start')
    logger.info('Config: toy_set = %s, top10_labels = %s, top100_labels = %s', toy_set, top10_labels, top100_labels)

    db = DatabaseManager()
    _, corpus = db.get_corpus(toy_set=toy_set, top10_labels=top10_labels, top100_labels=top100_labels)

    vocabulary_generator = VocabularyGenerator(corpus, logger)
    vocabulary = vocabulary_generator.build_vocabulary()

    filename = get_filename_vocabulary(time, toy_set, top10_labels, top100_labels)
    write_vocabulary_to_file(vocabulary, filename)
    logger.info('Vocabulary written to {}'.format(filename))
