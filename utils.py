from DatabaseManager import DatabaseManager


def get_icd9_codes_map(top10_labels=True, top100_labels=False):
    assert(top10_labels ^ top100_labels)

    db = DatabaseManager()
    icd9_codes = db.get_icd9_codes(top10_labels, top100_labels)
    icd9_codes_map = {code: i for i, code in enumerate(icd9_codes)}

    return icd9_codes_map