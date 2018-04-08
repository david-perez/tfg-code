SELECT COUNT(DISTINCT subject_id) FROM diagnoses_icd, top10_labels
  WHERE diagnoses_icd.icd9_code IN (top10_labels.icd9_code);

-- 32063

SELECT COUNT(DISTINCT subject_id) FROM diagnoses_icd
  WHERE diagnoses_icd.icd9_code IN
        ('4019',
         '4280',
         '42731',
         '41401',
         '5849',
         '25000',
         '2724',
         '51881',
         '5990',
         '53081');

-- 32063

SELECT COUNT(subject_id) FROM diagnoses_icd, top10_labels
  WHERE diagnoses_icd.icd9_code IN (top10_labels.icd9_code);

-- 106379

SELECT COUNT(subject_id) FROM diagnoses_icd
  WHERE diagnoses_icd.icd9_code IN
        ('4019',
         '4280',
         '42731',
         '41401',
         '5849',
         '25000',
         '2724',
         '51881',
         '5990',
         '53081');

-- 106379

WITH cartesian_product AS (
  SELECT
    row_id,
    subject_id,
    hadm_id,
    seq_num,
    diagnoses_icd.icd9_code AS diagnoses_icd_icd9_code,
    top10_labels.icd9_code,
    short_title,
    long_title,
    "Number of patients with label",
    "Percentage of patients with label"
  FROM diagnoses_icd, top10_labels
)
SELECT COUNT(*) FROM cartesian_product WHERE diagnoses_icd_icd9_code IN
        ('4019',
         '4280',
         '42731',
         '41401',
         '5849',
         '25000',
         '2724',
         '51881',
         '5990',
         '53081');

-- 1063790
