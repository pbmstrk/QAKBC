import logging

import numpy as np
from collections import OrderedDict
from sklearn.preprocessing import normalize

from .. import retriever
from ..reader import BatchReader



class ODQA:

    def __init__(self, reader, retriever, db):

        self.reader = reader
        self.retriever = retriever
        self.db = db

    def fetch_text(self, doc_id):
        return self.db.get_doc_text(doc_id)

    def process_query(self, query, topn=5, ndocs=10):
        ranked = [self.retriever.closest_docs(query, k=ndocs)]
        docids, docscores = zip(*ranked)
        docids, docscores = docids[0], docscores[0]

        # remove duplicates
        d = OrderedDict(list(zip(docids, docscores)))
        doc_texts = map(self.fetch_text, list(d.keys()))
        doc_texts = [text[0] for text in doc_texts]
        doc_scores = normalize(np.array(docscores)[:, np.newaxis], axis=0)[:, np.newaxis]

        batch = (query, doc_texts)
        span_scores = self.reader.predict(batch, topn=topn)
        scores = (1 - 0.7)*doc_scores + 0.7*span_scores
        inds = np.argpartition(scores, -topn, axis=None)[-topn:]
        inds = inds[np.argsort(np.take(scores, inds))][::-1]
        inds3d = zip(*np.unravel_index(inds, scores.shape))
        spans = self.reader.get_span(inds3d)
        final_scores = np.take(scores, inds)
        predictions = [{'span': spans[i], 'score': final_scores[i]} for i in range(len(spans))]
        return predictions
