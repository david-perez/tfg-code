CREATE MATERIALIZED VIEW patients_with_top10_labels
AS
  SELECT DISTINCT subject_id FROM diagnoses_icd, top10_labels
  WHERE diagnoses_icd.icd9_code IN (top10_labels.icd9_code)
WITH DATA;
