import os
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

    def get_corpus(self, toy_set=None, top100_labels=False, validation_set=False, test_set=False):
        cur = self.__conn.cursor()

        if test_set:
            dataset = 'test_set_'
        elif validation_set:
            dataset = 'validation_set_'
        else:
            dataset = 'training_set_'

        table_name = dataset + ('top100_labels' if top100_labels else 'top10_labels');
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

    def insert_bag_of_words_vectors(self, bag_of_words_vectors, vocabulary_filename, validation_set=False, test_set=False):
        cur = self.__conn.cursor()

        table_name = 'bw_'
        if test_set:
            table_name += 'test_'
        elif validation_set:
            table_name += 'validation_'
        vocabulary_filename, _ = os.path.splitext(vocabulary_filename)  # Remove extension.
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

    def get_icd9_codes(self, top100_labels=False, subject_id=None, validation_set=False, test_set=False):
        if top100_labels:
            assert (subject_id is None)

        cur = self.__conn.cursor()

        if subject_id is None:
            if top100_labels:
                pass
            else:
                cur.execute('SELECT icd9_code FROM top10_labels ORDER BY icd9_code ASC')
        else:
            if test_set:
                table_name = 'test_set'
            elif validation_set:
                table_name = 'validation_set'
            else:
                table_name = 'training_set'

            table_name += ('_top100_labels' if top100_labels else '_top10_labels') + '_patients_and_diagnoses'
            cur.execute('SELECT icd9_code FROM %s WHERE subject_id = %s ORDER BY icd9_code ASC', (AsIs(table_name), subject_id,))

        return [row[0] for row in cur.fetchall()]

    def get_patients_with_icd9_codes(self, icd9_codes):
        cur = self.__conn.cursor()

        cur.execute("""
          SELECT subject_id
          FROM diagnoses_icd_top10_from_patients_with_top10_labels
          GROUP BY subject_id
          HAVING array_agg(DISTINCT icd9_code) @> %s::varchar[]
        """, (icd9_codes,))

        return [row[0] for row in cur.fetchall()]

    def __close_connection(self):
        if self.__conn is not None:
            self.__conn.close()

    def get_bag_of_words_vectors(self, table_name):
        cur = self.__conn.cursor()
        cur.execute('SELECT * FROM %s ORDER BY subject_id ASC', (AsIs(table_name),))

        return cur
