from configparser import ConfigParser

import pickle
import psycopg2


class DatabaseManager:
    def __init__(self):
        self.__config = self.__load_config()
        self.__conn = self.__connect(self.__config)

    @staticmethod
    def __load_config(filename='database.ini', section='postgresql'):
        """ Return the configuration parameters contained in :param filename. """
        parser = ConfigParser()
        parser.read(filename)

        # Get section, default to postgresql.
        db = {}
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                db[param[0]] = param[1]
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, filename))

        return db

    @staticmethod
    def __connect(config):
        """ Return a connection to the PostgreSQL server. """
        conn = None
        try:
            conn = psycopg2.connect(**config)

            # Test the connection.
            cur = conn.cursor()
            cur.execute('SELECT version()')
            cur.fetchone()
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)

        return conn

    def get_corpus_training_set(self, toy_set=None, top10_labels=True, top100_labels=False):
        assert(top10_labels ^ top100_labels)

        cur = self.__conn.cursor()

        if top10_labels:
            if toy_set is None:
                cur.execute('SELECT * FROM training_set_top10_labels ORDER BY subject_id ASC, r ASC')
            else:
                cur.execute('SELECT * FROM training_set_top10_labels ORDER BY subject_id ASC, r ASC LIMIT %s', (toy_set,))
        elif top100_labels:
            pass

        subject_ids = []
        notes = []
        for row in cur:
            notes.append(row[5])
            subject_ids.append(row[1])

        return subject_ids, notes

    def insert_bag_of_words_vectors_training_set(self, bag_of_words_vectors, toy_set=None, top10_labels=True, top100_labels=False):
        assert(top10_labels ^ top100_labels)

        cur = self.__conn.cursor()

        if top10_labels:
            for subject_id, how_many_notes, bag_of_words_col_ind, bag_of_words_data in bag_of_words_vectors:
                cur.execute('INSERT INTO bag_of_words_training_set_top10_labels VALUES (%s, %s, %s, %s)',
                            (subject_id, how_many_notes,
                             psycopg2.Binary(pickle.dumps(bag_of_words_col_ind)),
                             psycopg2.Binary(pickle.dumps(bag_of_words_data))))
        elif top100_labels:
            pass

        self.__conn.commit()

    def get_icd9_codes(self, top10_labels=True, top100_labels=False, subject_id=None):
        assert((top10_labels ^ top100_labels) or (subject_id is not None))

        cur = self.__conn.cursor()

        if subject_id is None:
            if top10_labels:
                cur.execute('SELECT icd9_code FROM top10_labels ORDER BY icd9_code ASC')
            elif top100_labels:
                pass
        else:
            cur.execute("""SELECT icd9_code FROM training_set_top10_labels_patients_and_diagnoses WHERE subject_id = %s
                        ORDER BY icd9_code ASC""", (subject_id,))

        return [row[0] for row in cur.fetchall()]

    def __close_connection(self):
        if self.__conn is not None:
            self.__conn.close()

    def get_bag_of_words_vectors_training_set(self, toy_set=None, top10_labels=True, top100_labels=False):
        assert(top10_labels ^ top100_labels)

        cur = self.__conn.cursor()

        if top10_labels:
            if toy_set is None:
                cur.execute('SELECT * FROM bag_of_words_training_set_top10_labels ORDER BY subject_id ASC')
            else:
                cur.execute('SELECT * FROM bag_of_words_training_set_top10_labels ORDER BY subject_id ASC LIMIT %s',
                            (toy_set,))
        elif top100_labels:
            pass

        return cur
