CREATE MATERIALIZED VIEW top10_labels
AS
  WITH number_of_distinct_patients AS (
    SELECT COUNT(DISTINCT subject_id) AS "Number of distinct patients" FROM admissions
  ), count_of_each_icd9_code AS (
    SELECT icd9_code, COUNT(icd9_code) AS count
    FROM diagnoses_icd
    GROUP BY icd9_code
  )
  SELECT
    count_of_each_icd9_code.icd9_code,
    short_title,
    long_title,
    count AS "Number of patients with label",
    1.0 * count / "Number of distinct patients" AS "Percentage of patients with label"
  FROM
    number_of_distinct_patients,
    count_of_each_icd9_code
  INNER JOIN d_icd_diagnoses ON count_of_each_icd9_code.icd9_code = d_icd_diagnoses.icd9_code
  ORDER BY "Number of patients with label" DESC LIMIT 10
WITH DATA;

-- Top 10 ICD9 labels:
-- ['4019',
--  '4280',
--  '42731',
--  '41401',
--  '5849',
--  '25000',
--  '2724',
--  '51881',
--  '5990',
--  '53081']
