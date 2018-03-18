import pickle
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from DatabaseManager import DatabaseManager
from SpacyAnalyzer import SpacyAnalyzer
from timing import log


class BagOfWordsGenerator:
    def __init__(self, vocabulary, subject_ids, corpus):
        # corpus is a Python list containing medical notes.
        # The note corpus[i] belongs to the patient subject_ids[i].

        self.__vocabulary = vocabulary
        self.__corpus = corpus
        self.__subject_ids = subject_ids

        self.__analyzer = SpacyAnalyzer()

    def build_bag_of_words_vectors(self):
        vectorizer = CountVectorizer(analyzer=self.__analyzer.analyze, vocabulary=self.__vocabulary)
        X = vectorizer.fit_transform(self.__corpus)
        log("Term-document count matrix computed")

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

        return ret


def load_vocabulary():
    with open('vocabulary.p', 'rb') as fp:
        return pickle.load(fp)


if __name__ == '__main__':
    vocabulary = load_vocabulary()

    db = DatabaseManager()
    subject_ids, corpus = db.get_corpus_training_set(toy_set=None, top10_labels=True)
    # subject_ids, corpus = db.get_corpus_test_set(toy_set=None, top10_labels=True)

    bag_of_words_generator = BagOfWordsGenerator(vocabulary, subject_ids, corpus)
    bag_of_words_vectors = bag_of_words_generator.build_bag_of_words_vectors()

    db.insert_bag_of_words_vectors_training_set(bag_of_words_vectors, toy_set=None, top10_labels=True)
    # db.insert_bag_of_words_vectors_test_set(bag_of_words_vectors, toy_set=None, top10_labels=True)
