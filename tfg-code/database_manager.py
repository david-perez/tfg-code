import json
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
        chart_dates = []
        for row in cur:
            notes.append(row[5])
            subject_ids.append(row[1])
            chart_dates.append(row[2])

        return subject_ids, notes, chart_dates

    def insert_bag_of_words_vectors(self, bag_of_words_vectors, table_name):
        cur = self.__conn.cursor()

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

    def insert_bag_of_words_vectors_rnn(self, bag_of_words_vectors, table_name):
        cur = self.__conn.cursor()

        cur.execute("""
            CREATE TABLE %s (
                row_id BIGINT PRIMARY KEY NOT NULL,
                subject_id INTEGER,
                chart_date TIMESTAMP(0),
                bag_of_words_binary_vector_col_ind BYTEA NOT NULL,
                bag_of_words_binary_vector_data BYTEA NOT NULL
            );
        """, (AsIs(table_name),))
        self.__conn.commit()

        for row_id, (subject_id, chart_date, bag_of_words_col_ind, bag_of_words_data) in enumerate(bag_of_words_vectors):
            cur.execute('INSERT INTO %s VALUES (%s, %s, %s, %s, %s)',
                        (AsIs(table_name), row_id, subject_id, chart_date,
                         psycopg2.Binary(pickle.dumps(bag_of_words_col_ind)),
                         psycopg2.Binary(pickle.dumps(bag_of_words_data))))
        self.__conn.commit()

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

    def unique_subject_ids(self, table_name):
        cur = self.__conn.cursor()
        cur.execute('SELECT DISTINCT subject_id FROM %s ORDER BY subject_id ASC', (AsIs(table_name),))

        return [row[0] for row in cur.fetchall()]

    def get_bag_of_words_vectors_rnn(self, table_name, first_patient, last_patient):
        cur = self.__conn.cursor()

        cur.execute("""
          WITH append_note_in_seq AS (
          SELECT ROW_NUMBER()
          OVER (
            PARTITION BY subject_id
            ORDER BY chart_date ASC) AS note_in_seq,
            %s.*
            FROM %s
          ) SELECT * FROM append_note_in_seq
          WHERE %s <= subject_id AND subject_id <= %s
          ORDER BY subject_id ASC, chart_date ASC
        """, (AsIs(table_name), AsIs(table_name), first_patient, last_patient))

        return cur

    def vocabulary_experiment_create(self, config, start, comments=''):
        cur = self.__conn.cursor()
        cur.execute('INSERT INTO vocabulary_experiments (comments, config, start) VALUES(%s, %s, %s) RETURNING experiment_id',
                    (comments, json.dumps(config), start))
        self.__conn.commit()

        experiment_id = cur.fetchone()[0]
        return experiment_id

    def vocabulary_experiment_insert_log_file(self, experiment_id, log_filename):
        cur = self.__conn.cursor()
        cur.execute('UPDATE vocabulary_experiments SET log_filename = %s WHERE experiment_id = %s',
                    (log_filename, experiment_id))
        self.__conn.commit()

    def vocabulary_experiment_insert_vocabulary(self, experiment_id, end, vocabulary):
        cur = self.__conn.cursor()
        cur.execute('UPDATE vocabulary_experiments SET "end" = %s, vocabulary = %s WHERE experiment_id = %s',
                    (end, json.dumps(vocabulary), experiment_id))
        self.__conn.commit()

    def bag_of_words_generator_experiment_create(self, config, start, comments=''):
        cur = self.__conn.cursor()
        cur.execute(
            'INSERT INTO bag_of_words_generator_experiments (comments, config, start) VALUES(%s, %s, %s) RETURNING experiment_id',
            (comments, json.dumps(config), start))
        self.__conn.commit()

        experiment_id = cur.fetchone()[0]
        return experiment_id

    def bag_of_words_generator_experiment_insert_log_file(self, experiment_id, log_filename):
        cur = self.__conn.cursor()
        cur.execute('UPDATE bag_of_words_generator_experiments SET log_filename = %s WHERE experiment_id = %s',
                    (log_filename, experiment_id))
        self.__conn.commit()

    def load_vocabulary(self, vocabulary_experiment):
        cur = self.__conn.cursor()
        cur.execute('SELECT vocabulary FROM vocabulary_experiments WHERE experiment_id = %s', (vocabulary_experiment,))

        return cur.fetchone()[0]

    def bag_of_words_generator_experiment_insert_table_name(self, experiment_id, end, table_name):
        cur = self.__conn.cursor()
        cur.execute('UPDATE bag_of_words_generator_experiments SET "end" = %s, table_name = %s WHERE experiment_id = %s',
                    (end, table_name, experiment_id))
        self.__conn.commit()

    def classifier_experiment_create(self, config, start, classifier_name, train_table_name, val_table_name, test_table_name, comments=''):
        cur = self.__conn.cursor()
        cur.execute("""
            INSERT INTO classifier_experiments
            (classifier_name, comments, config, start, train_table_name, val_table_name, test_table_name)
            VALUES(%s, %s, %s, %s, %s, %s, %s) RETURNING experiment_id
        """, (classifier_name, comments, json.dumps(config), start, train_table_name, val_table_name, test_table_name))
        self.__conn.commit()

        experiment_id = cur.fetchone()[0]
        return experiment_id

    def classifier_experiment_insert_log_file(self, experiment_id, log_filename):
        cur = self.__conn.cursor()
        cur.execute('UPDATE classifier_experiments SET log_filename = %s WHERE experiment_id = %s',
                    (log_filename, experiment_id))
        self.__conn.commit()

    def classifier_experiment_insert_metrics(self, experiment_id, metrics_train, metrics_val, metrics_test, end):
        cur = self.__conn.cursor()
        cur.execute("""
            UPDATE classifier_experiments
            SET metrics_train = %s, metrics_val = %s, metrics_test = %s, "end" = %s
            WHERE experiment_id = %s
        """, (json.dumps(metrics_train), json.dumps(metrics_val), json.dumps(metrics_test), end, experiment_id))
        self.__conn.commit()

    def get_patients_with_number_icd9_codes(self):
        cur = self.__conn.cursor()
        cur.execute('SELECT subject_id, COUNT(DISTINCT icd9_code) AS cnt FROM diagnoses_icd GROUP BY subject_id ORDER BY cnt DESC')
        return cur

    def get_icd9_codes_distribution(self):
        cur = self.__conn.cursor()
        cur.execute('SELECT icd9_code, COUNT(icd9_code) AS cnt FROM diagnoses_icd GROUP BY icd9_code ORDER BY cnt DESC')
        return cur

    def get_patients_with_number_of_notes_with_top10_labels(self):
        cur = self.__conn.cursor()
        cur.execute('SELECT subject_id, COUNT(*) AS cnt FROM noteevents_from_patients_with_top10_labels GROUP BY subject_id ORDER BY cnt DESC')
        return cur

    def get_corpus_all_splits(self):
        cur = self.__conn.cursor()
        cur.execute('SELECT text FROM filtered_pruned_dataset_top10_labels;')
        return cur
