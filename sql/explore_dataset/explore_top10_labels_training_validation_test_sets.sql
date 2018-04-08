-- Let's build the training, validation and test sets.
-- 31865 / 2 = 15932
-- 15932 / 2 = 7966
-- Training set: patients in rows 1-15932 ==> 15932 rows.
-- Validation set: patients in rows 15933-23898 ==> 7966 rows.
-- Test set: patients in rows 23899-31865 ==> 7967 rows.

SELECT MAX(subject_id) FROM (
    SELECT DISTINCT ON (subject_id) r, subject_id, chartdate
    FROM filtered_pruned_dataset_top10_labels
    ORDER BY subject_id, r ASC
    LIMIT 15932
) AS max_subject_id_training_set;

-- 27953

SELECT MAX(subject_id) FROM (
    SELECT DISTINCT ON (subject_id) r, subject_id, chartdate
    FROM filtered_pruned_dataset_top10_labels
    ORDER BY subject_id, r ASC
    LIMIT 7966 OFFSET 15932
) AS max_subject_id_validation_set;

-- 62774

SELECT MAX(subject_id) FROM (
    SELECT DISTINCT ON (subject_id) r, subject_id, chartdate
    FROM filtered_pruned_dataset_top10_labels
    ORDER BY subject_id, r ASC
    LIMIT 7967 OFFSET 15932 + 7966
) AS max_subject_id_test_set;

-- 99999

-- We can now build the training, validation and test sets using subject_id.
-- Training set: patients with subject_ids 1-27953 ==> 15932 patients.
-- Validation set: patients with subject_ids 27954-62774 ==> 7966 patients.
-- Test set: patients in with subject_ids 62775-99999 ==> 7967 patients.

SELECT COUNT(DISTINCT subject_id) FROM (
    SELECT r, subject_id, chartdate, category, description, text
    FROM filtered_pruned_dataset_top10_labels WHERE subject_id BETWEEN 1 AND 27953
    ORDER BY subject_id ASC
) AS number_patients_training_set;

-- 15932

SELECT COUNT(DISTINCT subject_id) FROM (
    SELECT r, subject_id, chartdate, category, description, text
    FROM filtered_pruned_dataset_top10_labels WHERE subject_id BETWEEN 27954 AND 62774
    ORDER BY subject_id ASC
) AS number_patients_validation_set;

-- 7966

SELECT COUNT(DISTINCT subject_id) FROM (
    SELECT r, subject_id, chartdate, category, description, text
    FROM filtered_pruned_dataset_top10_labels WHERE subject_id BETWEEN 62775 AND 99999
    ORDER BY subject_id ASC
) AS number_patients_test_set;

-- 7967

-- Let's see if the notes are more or less distributed in the correct proportions across the training, validation and test sets.

SELECT COUNT(*) FROM training_set_top10_labels;

-- 269147

SELECT COUNT(*) FROM validation_set_top10_labels;

-- 130371

SELECT COUNT(*) FROM test_set_top10_labels;

-- 125237

-- 269147 + 130371 + 125237 = 524755 ✔️