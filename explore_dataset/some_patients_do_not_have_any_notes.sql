SELECT COUNT(DISTINCT subject_id) FROM diagnoses_icd_top10_from_patients_with_top10_labels;

-- 32063 (Warning: there are patients in diagnoses_icd_top10_from_patients_with_top10_labels who don't have any notes).

SELECT COUNT(*) FROM noteevents, patients_with_top10_labels
WHERE noteevents.subject_id=patients_with_top10_labels.subject_id;

-- 1522558

-- ¿Are there patients who don't have any notes?

SELECT COUNT(DISTINCT patients_with_top10_labels.subject_id) FROM patients_with_top10_labels
INNER JOIN noteevents ON patients_with_top10_labels.subject_id = noteevents.subject_id;

-- 31865

SELECT COUNT(DISTINCT patients_with_top10_labels.subject_id) FROM patients_with_top10_labels
LEFT JOIN noteevents ON patients_with_top10_labels.subject_id = noteevents.subject_id;

-- 32063 (Yes, there are patients who don't have any notes)

WITH patients_with_top10_labels_who_dont_have_notes AS (
    SELECT
      patients_with_top10_labels.subject_id AS "p_subject_id",
      noteevents.subject_id AS "n_subject_id"
    FROM patients_with_top10_labels
      LEFT JOIN noteevents ON patients_with_top10_labels.subject_id = noteevents.subject_id
    WHERE noteevents.subject_id IS NULL
) SELECT COUNT(DISTINCT patients_with_top10_labels_who_dont_have_notes.p_subject_id) FROM patients_with_top10_labels_who_dont_have_notes;

-- 198 (i.e. 32063 - 31865)
