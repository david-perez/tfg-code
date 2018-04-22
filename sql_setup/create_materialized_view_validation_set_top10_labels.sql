CREATE MATERIALIZED VIEW validation_set_top10_labels AS
    SELECT r, subject_id, chartdate, category, description, text
    FROM filtered_pruned_dataset_top10_labels WHERE subject_id BETWEEN 27954 AND 62774
    ORDER BY subject_id ASC
WITH DATA;
