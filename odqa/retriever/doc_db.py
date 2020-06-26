# this code is adapted from original source code at
# https://github.com/facebookresearch/DrQA

# The License file in the root of the directory is the original
# license file

# code to access .db file containing wikipedia dump


import sqlite3
from . import utils


class DocDB:
    """
    Sqlite backed document storage.
    """

    def __init__(self, db_path):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)

    @property
    def path(self):
        """Return the path to the file that backs this database."""
        return self.db_path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_text(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT text FROM documents WHERE id = ?",
            (utils.normalize(doc_id),)
        )
        result = cursor.fetchone()
        cursor.close()
        return result
