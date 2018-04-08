CREATE MATERIALIZED VIEW filtered_pruned_dataset_top10_labels
AS
  WITH noteevents_from_patients_with_top10_labels AS (
    SELECT
      noteevents.subject_id,
      chartdate,
      category,
      description,
      text
    FROM noteevents, patients_with_top10_labels
    WHERE noteevents.subject_id=patients_with_top10_labels.subject_id
  )
  SELECT * FROM (
    SELECT
      ROW_NUMBER()
      OVER (
        PARTITION BY subject_id
        ORDER BY chartdate DESC) AS r,
      noteevents_from_patients_with_top10_labels.*
    FROM noteevents_from_patients_with_top10_labels
  ) x
  WHERE x.r <= 20
WITH DATA;
