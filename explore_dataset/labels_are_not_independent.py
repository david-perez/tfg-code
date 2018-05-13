import itertools
import sys

"""
Disease labels are not independent from each other in the dataset. Some pairs of labels occur more frequently than
others. 

Execute this script from the project root folder.
"""

sys.path.append('tfg-code/')

from DatabaseManager import DatabaseManager

db = DatabaseManager()
codes = db.get_icd9_codes()

l = []
for x, y in itertools.combinations(codes, 2):
    x_cnt = len(db.get_patients_with_icd9_codes([x]))
    y_cnt = len(db.get_patients_with_icd9_codes([y]))
    xy_cnt = len(db.get_patients_with_icd9_codes([x, y]))
    l.append((x, y, x_cnt, y_cnt, xy_cnt, xy_cnt / x_cnt, xy_cnt / y_cnt))

r1 = sorted(l, key=lambda tup: tup[5], reverse=True)[0]
r2 = sorted(l, key=lambda tup: tup[6], reverse=True)[0]

if r1[5] > r2[6]:
    r = r1
else:
    r = r2

x, y, x_cnt, y_cnt, xy_cnt, xy_x, xy_y = r
print('{} patients have disease {}. {} patients have disease {}.'.format(x_cnt, x, y_cnt, y))
print('{} patients have both.'.format(xy_cnt))
print('{} of patients who have disease {} have also have disease {}.'.format(xy_y, y, x))
print('{} of patients who have disease {} have also disease {}.'.format(xy_x, x, y))
