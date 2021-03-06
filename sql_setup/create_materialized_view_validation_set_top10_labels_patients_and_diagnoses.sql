CREATE MATERIALIZED VIEW validation_set_top10_labels_patients_and_diagnoses AS (
    WITH patients_validation_set_with_top10_labels_who_have_notes AS (
      SELECT DISTINCT subject_id
      FROM validation_set_top10_labels
      ORDER BY subject_id ASC
    )
    SELECT patients_validation_set_with_top10_labels_who_have_notes.subject_id, diagnoses_icd.icd9_code
    FROM patients_validation_set_with_top10_labels_who_have_notes
    INNER JOIN diagnoses_icd ON patients_validation_set_with_top10_labels_who_have_notes.subject_id = diagnoses_icd.subject_id
    INNER JOIN top10_labels ON diagnoses_icd.icd9_code = top10_labels.icd9_code
    ORDER BY patients_validation_set_with_top10_labels_who_have_notes.subject_id ASC
) WITH DATA;
