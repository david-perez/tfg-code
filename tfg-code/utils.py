from database_manager import DatabaseManager


def get_icd9_codes_map(top100_labels=False):
    db = DatabaseManager()
    icd9_codes = db.get_icd9_codes(top100_labels=top100_labels)
    icd9_codes_map = {code: i for i, code in enumerate(icd9_codes)}

    return icd9_codes_map
