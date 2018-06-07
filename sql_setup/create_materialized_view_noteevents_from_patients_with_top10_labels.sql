CREATE MATERIALIZED VIEW noteevents_from_patients_with_top10_labels AS
SELECT
      noteevents.subject_id,
      chartdate,
      category,
      description,
      text
    FROM noteevents, patients_with_top10_labels
    WHERE noteevents.subject_id=patients_with_top10_labels.subject_id
WITH DATA;
