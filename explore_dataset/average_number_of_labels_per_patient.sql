WITH count_patients AS (
  SELECT COUNT(*) FROM patients AS count
), count_all_labels AS (
  SELECT COUNT(*) FROM diagnoses_icd AS count
)
SELECT 1.0 * count_all_labels.count / count_patients.count AS "Average number of labels per patient" FROM count_all_labels, count_patients;

-- 13.9949914015477214
