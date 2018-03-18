import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from DatabaseManager import DatabaseManager
from SpacyAnalyzer import SpacyAnalyzer
from timing import log


class VocabularyGenerator:
    def __init__(self, corpus, filename, tf_idf_cutoff=40000, word_appearance_in_documents_min_threshold=3):
        self.__corpus = corpus
        self.__filename = filename
        self.__tf_idf_cutoff=tf_idf_cutoff

        self.__analyzer = SpacyAnalyzer()

        # Consider words that appear in at least word_appearance_in_documents_min_threshold documents.
        self.__word_appearance_in_documents_min_threshold = word_appearance_in_documents_min_threshold

    def __build_vocabulary_word_appearance_in_documents(self):
        vectorizer = CountVectorizer(analyzer=self.__analyzer.analyze)
        X = vectorizer.fit_transform(self.__corpus)
        log("Term-document count matrix computed")

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
        log('Term-document TF-IDF matrix computed')

        sum_columns = np.asarray(np.sum(X, axis=0))[0]

        # The vocabulary is selected as the top cutoff terms ordered by TF-IDF scores.
        # We first sort the TF-IDF scores in descending order (reverse=True) and get the indices of the feature_names of
        # the first cutoff terms. With those indices, we look into the feature_names array to obtain the vocabulary.
        feature_names = vectorizer.get_feature_names()
        vocabulary = [feature_names[idx] for idx, _ in
                      sorted(enumerate(sum_columns), key=lambda t: t[1], reverse=True)[:self.__tf_idf_cutoff]]
        log('Final vocabulary calculated')

        return vocabulary

    def __write_vocabulary_to_file(self, vocabulary):
        with open(self.__filename, 'wb') as fp:
            pickle.dump(vocabulary, fp)
        log('Vocabulary written to {}'.format(self.__filename))

    def build_vocabulary(self):
        vocabulary = self.__build_vocabulary_word_appearance_in_documents()
        print('There are {} words that appear in at least {} documents.'
              .format(len(vocabulary), self.__word_appearance_in_documents_min_threshold))

        vocabulary = self.__build_vocabulary_tf_idf_scores(vocabulary)
        print('The first {} terms (ordered by TF-IDF scores) that appear in at least {} documents of the training set are:'
            .format(self.__tf_idf_cutoff, self.__word_appearance_in_documents_min_threshold))
        print(vocabulary)

        self.__write_vocabulary_to_file(vocabulary)

        return vocabulary


if __name__ == '__main__':
    db = DatabaseManager()
    _, corpus = db.get_corpus_training_set(toy_set=None, top10_labels=True)

    vocabulary_generator = VocabularyGenerator(corpus, 'vocabulary.p')
    vocabulary = vocabulary_generator.build_vocabulary()
