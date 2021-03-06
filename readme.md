# tfg-code

## Installation

Run the following SQL scripts in order.

```
create_materialized_view_top10_labels.sql
create_materialized_view_patients_with_top10_labels.sql
create_materialized_view_diagnoses_icd_top10_from_patients_with_top10_labels.sql
create_materialized_view_filtered_pruned_dataset_top10_labels.sql #  (takes ~15 minutes)
create_materialized_view_training_set_top10_labels.sql
create_materialized_view_validation_set_top10_labels.sql
create_materialized_view_test_set_top10_labels.sql
create_materialized_view_training_set_top10_labels_patients_and_diagnoses.sql
create_materialized_view_test_set_top10_labels_patients_and_diagnoses.sql
create_tables_experiments.sql
create_materialized_view_noteevents_from_patients_with_top10_labels.sql
```

Some valuable insights into the dataset can be gained by running the queries and scripts in the folder `explore_dataset/`.

Install the project's dependencies using `pip install -r requirements.txt`.

Download the `en_core_web_sm` spaCy model and symlink it to `en` within the `spacy/data` directory. Both of these things
can be done using the command [`python -m spacy download en`](https://spacy.io/models/).

## Setup

Fill in the connection details to a PostgreSQL database containing the [MIMIC-III](https://mimic.physionet.org/) dataset
in a file named `database.ini` (see the example file `database.ini.example`).

## Usage

All scripts must be run from the `tfg-code` directory.

`vocabulary_generator.py` generates the vocabulary from the corpora in the training set. It outputs a json file containing
a list of words that is serialized to the `vocabulary_experiments` table.

`bag_of_words_generator.py` generates bag of words for each patient in the training set or in the test set, using a provided
vocabulary. It creates a table in the database containing the serialized (in binary) bag of words vectors for each patient.
For RNNs, it creates bag of words vectors for each medical note. Experiment results are output to the `bag_of_words_generator_experiments`
table.

When running models that log to TensorBoard (`feed_forward_nn.py` and `rnn.py`), ensure TensorBoard is run using

```
tensorboard --logdir=tensorboard_logs/
``` 

### Models

`logistic_regression.py` reads bag of words vectors from a training set and a test set, stored in the provided tables,
and evaluates the performance of a collection of logistic regression classifiers, one for each ICD-9 label.

`feed_forward_nn.py` and `rnn.py` run neural network models using bag of words vectors from tables in the database
provided as arguments to the program.

Experiment metrics are stored in the `classifier_experiments` table.

All log files are stored in the `logs` directory.

Use the option `--toy_set` to try out the system with a reduced dataset. All program arguments and flags can be displayed
passing the `--help` option to the scripts.
