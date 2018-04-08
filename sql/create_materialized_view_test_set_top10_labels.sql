CREATE MATERIALIZED VIEW test_set_top10_labels AS
    SELECT r, subject_id, chartdate, category, description, text
    FROM filtered_pruned_dataset_top10_labels WHERE subject_id BETWEEN 62775 AND 99999
    ORDER BY subject_id ASC
WITH DATA;