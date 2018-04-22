CREATE MATERIALIZED VIEW diagnoses_icd_top10_from_patients_with_top10_labels
AS
  SELECT row_id, diagnoses_icd.subject_id, hadm_id, seq_num, diagnoses_icd.icd9_code
  FROM diagnoses_icd, top10_labels, patients_with_top10_labels
  WHERE diagnoses_icd.subject_id IN (patients_with_top10_labels.subject_id)
  AND diagnoses_icd.icd9_code IN (top10_labels.icd9_code)
WITH DATA;
