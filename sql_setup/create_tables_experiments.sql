CREATE TABLE vocabulary_experiments (
    experiment_id SERIAL PRIMARY KEY,
    comments TEXT,
    config JSONB NOT NULL,
    start TIMESTAMP(0) NOT NULL,
    "end" TIMESTAMP(0),
    log_filename TEXT,
    vocabulary JSONB
);

CREATE TABLE bag_of_words_generator_experiments (
    experiment_id SERIAL PRIMARY KEY,
    comments TEXT,
    config JSONB NOT NULL,
    start TIMESTAMP(0) NOT NULL,
    "end" TIMESTAMP(0),
    log_filename TEXT,
    table_name TEXT
);

CREATE TABLE classifier_experiments (
    experiment_id SERIAL PRIMARY KEY,
    classifier_name TEXT,
    comments TEXT,
    config JSONB NOT NULL,
    start TIMESTAMP(0) NOT NULL,
    "end" TIMESTAMP(0),
    log_filename TEXT,
    metrics_train JSONB,
    metrics_val JSONB,
    metrics_test JSONB,
    train_table_name TEXT,
    val_table_name TEXT,
    test_table_name TEXT
);
