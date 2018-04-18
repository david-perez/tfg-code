WITH distribution_top10_labels_training_set AS (
  SELECT icd9_code, (1.0 * COUNT(icd9_code)) / (SELECT COUNT(*) FROM training_set_top10_labels_patients_and_diagnoses) AS "Percentage training"
  FROM training_set_top10_labels_patients_and_diagnoses
  GROUP BY icd9_code
), distribution_top10_labels_test_set AS (
  SELECT icd9_code, (1.0 * COUNT(icd9_code)) / (SELECT COUNT(*) FROM test_set_top10_labels_patients_and_diagnoses) AS "Percentage test"
  FROM test_set_top10_labels_patients_and_diagnoses
  GROUP BY icd9_code
), distribution_top10_labels_all_patients AS (
  SELECT icd9_code, (1.0 * COUNT(icd9_code)) / (SELECT COUNT(*) FROM diagnoses_icd_top10_from_patients_with_top10_labels) AS "Percentage all"
  FROM diagnoses_icd_top10_from_patients_with_top10_labels
  GROUP BY icd9_code
)
SELECT
  distribution_top10_labels_all_patients.icd9_code,
  ROUND(distribution_top10_labels_all_patients."Percentage all", 5) AS "Percentage all",
  ROUND(distribution_top10_labels_training_set."Percentage training", 5) AS "Percentage training",
  ROUND(distribution_top10_labels_test_set."Percentage test", 5) AS "Percentage test",
  ROUND(abs(distribution_top10_labels_all_patients."Percentage all" - distribution_top10_labels_training_set."Percentage training"), 5) AS "diff(all - train)",
  ROUND(abs(distribution_top10_labels_all_patients."Percentage all" - distribution_top10_labels_test_set."Percentage test"), 5) AS "diff(all - test)"
FROM distribution_top10_labels_all_patients
INNER JOIN distribution_top10_labels_training_set
  ON distribution_top10_labels_all_patients.icd9_code = distribution_top10_labels_training_set.icd9_code
INNER JOIN distribution_top10_labels_test_set
  ON distribution_top10_labels_all_patients.icd9_code = distribution_top10_labels_test_set.icd9_code
ORDER BY distribution_top10_labels_all_patients."Percentage all" DESC;

SELECT COUNT(*) FROM diagnoses_icd_top10_from_patients_with_top10_labels;
