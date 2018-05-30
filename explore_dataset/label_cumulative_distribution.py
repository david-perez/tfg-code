import sys

"""
Get label distribution.

Execute this script from the project root folder.
"""

sys.path.append('tfg-code/')

from database_manager import DatabaseManager

db = DatabaseManager()
rows = db.get_icd9_codes_distribution()

cum_sum = []
for icd9_code, cnt in rows:
    cum_sum.append(cnt)
    if len(cum_sum) > 1:
        cum_sum[-1] += cum_sum[-2]

print(cum_sum)