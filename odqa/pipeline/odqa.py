import logging
from .. import reader
from .. import retriever

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


def fetch_text(doc_id)


class ODQA:

    def __init__(self, reader_model, retriever_path, doc_db_path,  device):
        
        self.reader_model = reader_model
        self.retriever_path = retriever_path
        
        logger.info('Initializing document ranker...')
        self.retriever = TfidfDocRanker(self.retriever_path)
        self.reader = BatchReader(self.reader_model, device) 
        self.db = DocDB(doc_db_path)

    def fetch_text(self, doc_id):
        return self.db.get_doc_text(doc_id)

    def process_query(query, topn=5, ndocs=10):
        ranked = self.retriever.closest_docs(query, k=ndocs)
        
        docids, docscores = zip(*ranked)
        # remove duplicate duplicates
        flat_docids = list({d for docids in all_docids for d in docids})
        doc_texts = map(fetch_text, flat_docids)

        batch = ((query, context) for context in doc_texts))

        predictions = self.reader.predict(batch, topn=topn)
        return predictions

        

