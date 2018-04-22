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
```

Some valuable insights into the dataset can be gained by running the queries and scripts in the folder `explore_dataset/`.

Install the project's dependencies using `pip install -r requirements.txt`.

Download the `en_core_web_sm` spaCy model and symlink it to `en` within the `spacy/data` directory. Both of these things
can be done using the command [`python -m spacy download en`](https://spacy.io/models/).

## Setup

Fill in the connection details to a PostgreSQL database containing the [MIMIC-III](https://mimic.physionet.org/) dataset
in a file named `database.ini` (see the example file `database.ini.example`).

## Usage

`VocabularyGenerator.py` generates the vocabulary from the corpora in the training set. It outputs a json file containing
a list of words to the `serialized_vocabularies` directory.

`BagOfWordsGenerator.py` generates bag of words for each patient in the training set or in the test set, using a provided
vocabulary. It creates a table in the database containing the serialized (in binary) bag if words vectors for each patient.
The script outputs the name of the table that is created.

When running models that log to TensorBoard, ensure TensorBoard is run using

```
tensorboard --logdir=tensorboard_logs/
``` 

### Models

`LogisticRegression.py` reads bag of words vectors from a training set and a test set, stored in the provided tables,
and evaluates the performance of a collection of logistic regression classifiers, one for each ICD9 label.

## Example pipeline

The following is an example shell file that shows how the vocabulary generator, bag of words generator and a
classification model (logistic regression) can be used together.

```
#!/bin/bash

vocabulary_filename=$(python VocabularyGenerator.py --toy_set)
table_name_train=$(python BagOfWordsGenerator.py $vocabulary_filename --toy_set | tail -1)
table_name_test=$(python BagOfWordsGenerator.py $vocabulary_filename --toy_set --test_set | tail -1)

python LogisticRegression.py $table_name_train $table_name_test

```