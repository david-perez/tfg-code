from configparser import ConfigParser
from psycopg2.extensions import AsIs

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

    def get_corpus(self, toy_set=None, top100_labels=False, test_set=False):
        cur = self.__conn.cursor()

        table_name = ('test_set_' if test_set else 'training_set_') + ('top100_labels' if top100_labels else 'top10_labels');
        sql_select = 'SELECT * FROM %s ORDER BY subject_id ASC, r ASC'
        if toy_set is not None:
            sql_select += ' LIMIT %s'
            cur.execute(sql_select, (AsIs(table_name), toy_set))
        else:
            cur.execute(sql_select, (AsIs(table_name),))

        subject_ids = []
        notes = []
        for row in cur:
            notes.append(row[5])
            subject_ids.append(row[1])

        return subject_ids, notes

    def insert_bag_of_words_vectors(self, bag_of_words_vectors, vocabulary_filename, test_set=False):
        cur = self.__conn.cursor()

        table_name = 'bw_'
        if test_set:
            table_name += 'test_'
        table_name += vocabulary_filename.replace('.', '')
        cur.execute("""
            CREATE TABLE %s (
                subject_id INTEGER PRIMARY KEY NOT NULL,
                how_many_notes INTEGER NOT NULL,
                bag_of_words_binary_vector_col_ind BYTEA NOT NULL,
                bag_of_words_binary_vector_data BYTEA NOT NULL
            );
        """, (AsIs(table_name),))
        self.__conn.commit()

        for subject_id, how_many_notes, bag_of_words_col_ind, bag_of_words_data in bag_of_words_vectors:
            cur.execute('INSERT INTO %s VALUES (%s, %s, %s, %s)',
                        (AsIs(table_name), subject_id, how_many_notes,
                         psycopg2.Binary(pickle.dumps(bag_of_words_col_ind)),
                         psycopg2.Binary(pickle.dumps(bag_of_words_data))))

        self.__conn.commit()

        return table_name

    def get_icd9_codes(self, top10_labels=True, top100_labels=False, subject_id=None, test_set=False):
        assert((top10_labels ^ top100_labels) or (subject_id is not None))

        cur = self.__conn.cursor()

        if subject_id is None:
            if top10_labels:
                cur.execute('SELECT icd9_code FROM top10_labels ORDER BY icd9_code ASC')
            elif top100_labels:
                pass
        else:
            table_name = ('test_set' if test_set else 'training_set') + ('_top10_labels' if top10_labels else '_top100_labels') + '_patients_and_diagnoses'
            cur.execute('SELECT icd9_code FROM %s WHERE subject_id = %s ORDER BY icd9_code ASC', (AsIs(table_name), subject_id,))

        return [row[0] for row in cur.fetchall()]

    def __close_connection(self):
        if self.__conn is not None:
            self.__conn.close()

    def get_bag_of_words_vectors(self, table_name):
        cur = self.__conn.cursor()
        cur.execute('SELECT * FROM %s ORDER BY subject_id ASC', (AsIs(table_name),))

        return cur
