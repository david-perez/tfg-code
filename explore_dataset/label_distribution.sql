SELECT subject_id, COUNT(DISTINCT icd9_code) AS cnt FROM diagnoses_icd GROUP BY subject_id ORDER BY cnt DESC;

WITH label_distribution AS (
    SELECT
      icd9_code,
      COUNT(icd9_code) AS cnt
    FROM diagnoses_icd
    GROUP BY icd9_code
    ORDER BY cnt DESC
) SELECT COUNT(*) FROM label_distribution WHERE cnt < 5;
