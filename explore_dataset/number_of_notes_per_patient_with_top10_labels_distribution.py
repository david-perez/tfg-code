import sys

"""
Get distribution of number of notes per patient, where patients have been diagnosed with a top10 ICD-9 label.

Execute this script from the project root folder.
"""

sys.path.append('tfg-code/')

from database_manager import DatabaseManager

db = DatabaseManager()
rows = db.get_patients_with_number_of_notes_with_top10_labels()

N = 75

buckets = [0 for _ in range(N + 2)]

for _, number_of_diagnoses in rows:
    if number_of_diagnoses <= N:
        buckets[number_of_diagnoses] += 1
    elif N < number_of_diagnoses:
        buckets[-1] += 1


print(buckets)