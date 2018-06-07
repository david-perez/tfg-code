import sys

"""
Get distribution of note length from notes of the filtered pruned dataset.

Execute this script from the project root folder.
"""

sys.path.append('tfg-code/')

from database_manager import DatabaseManager

db = DatabaseManager()
rows = db.get_corpus_all_splits()

bucket_char_size = 40

minl, maxl = 1e9, 0
for text, in rows:
    minl = min(len(text), minl)
    maxl = max(len(text), maxl)

print('Min length: {}, Max length: {}'.format(minl, maxl))

n_buckets = (maxl // bucket_char_size) + 1
buckets = [0 for _ in range(n_buckets)]

rows = db.get_corpus_all_splits()
for text, in rows:
    idx = len(text) // bucket_char_size
    buckets[idx] += 1

print(buckets)
