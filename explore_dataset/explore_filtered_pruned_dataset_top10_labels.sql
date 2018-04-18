SELECT COUNT(*) FROM filtered_pruned_dataset_top10_labels;

-- 524755

SELECT COUNT(DISTINCT subject_id) FROM filtered_pruned_dataset_top10_labels;

-- 31865

SELECT DISTINCT ON (subject_id) r, subject_id, chartdate
FROM filtered_pruned_dataset_top10_labels
ORDER BY subject_id, r ASC;
