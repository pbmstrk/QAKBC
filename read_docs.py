import argparse
import json
import os
import logging
import torch

import numpy as np
from functools import partial
from tqdm import tqdm
from torch.utils import data
from collections import OrderedDict
from sklearn.preprocessing import normalize

from odqa.logger import set_logger
from odqa.reader import BatchReader
from odqa.retriever import DocDB

parser = argparse.ArgumentParser()
parser.add_argument('docs', type=str)
parser.add_argument('outdir', type=str)
parser.add_argument('readerpath', type=str)
parser.add_argument('dbpath', type=str)
parser.add_argument('--topn', type=int, default=10)
parser.add_argument('--logfile', type=str, default='./read_docs.log')


class ReaderDataset(data.Dataset):

    def __init__(self, queries, docids, docscores):
        self.queries = queries
        self.docids = docids
        self.docscores = docscores

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        X = self.queries[idx]
        Y = self.docids[idx]
        Z = self.docscores[idx]
        return X, Y, Z


def generate_batch(batch, db):

    query = batch[0][0]
    docids = batch[0][1]
    docscores = batch[0][2]

    d = OrderedDict(list(zip(docids, docscores)))
    doctexts = map(partial(fetch_text, db=db), list(d.keys()))
    doctexts = [text[0] for text in doctexts]
    docscores = normalize(np.array(docscores)[:, np.newaxis],
                          axis=0)[:, np.newaxis]

    batch = (query, doctexts, docscores)

    return batch


def initialise(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reader = BatchReader(args.readerpath, device)

    db = DocDB(args.dbpath)

    logging.info('Finished initialisation')

    return reader, db


def fetch_text(doc_id, db):
    return db.get_doc_text(doc_id)


if __name__ == '__main__':

    # parse command line arguments
    args = parser.parse_args()

    # set up logger
    set_logger(args.logfile)

    # initialise reader and DocDB
    reader, db = initialise(args)

    basename = os.path.splitext(os.path.basename(args.docs))[0]
    outfile = os.path.join(args.outdir, basename + '-' +
                           os.path.basename(args.readerpath) + '.preds')

    logging.info('Output file: {}'.format(outfile))

    logging.info("Retrieving data")

    all_doc_ids = []
    all_doc_scores = []
    queries = []

    for line in open(args.docs):
        dat = json.loads(line)
        all_doc_ids.append(dat['doc_ids'])
        all_doc_scores.append(dat['doc_scores'])
        queries.append(dat['query'])

    logging.info("Reading..")

    collate_fn = partial(generate_batch, db=db)

    querydataset = ReaderDataset(queries, all_doc_ids, all_doc_scores)
    data_generator = data.DataLoader(querydataset, batch_size=1,
                                     collate_fn=collate_fn, num_workers=0)

    with open(outfile, 'w') as f:
#        for i in tqdm(range(len(all_doc_ids))):
#            docids, docscores = all_doc_ids[i], all_doc_scores[i]
#            query = queries[i]
#
#            d = OrderedDict(list(zip(docids, docscores)))
#            doc_texts = map(partial(fetch_text, db=db), list(d.keys()))
#            doc_texts = [text[0] for text in doc_texts]
#            doc_scores = normalize(np.array(docscores)[:, np.newaxis],axis=0)[:, np.newaxis]

        for batch in tqdm(data_generator, total=len(queries)):
            query, doc_texts, doc_scores = batch
            span_scores = reader.predict((query, doc_texts), topn=args.topn)
            scores = (1 - 0.7)*doc_scores + 0.7*span_scores
            # inds = np.argpartition(scores, -args.topn, axis=None)[-args.topn:]
            # inds = inds[np.argsort(np.take(scores, inds))][::-1]
            # inds3d = zip(*np.unravel_index(inds, scores.shape))
            idx = scores.reshape(scores.shape[0], -1).argmax(-1)
            inds = list((np.arange(scores.shape[0]), *np.unravel_index(idx,
                         scores.shape[-2:])))
            inds = np.ravel_multi_index(inds, dims=scores.shape)
            inds = inds[np.argsort(np.take(scores, inds))][::-1][:args.topn]
            inds3d = zip(*np.unravel_index(inds, scores.shape))
            spans = reader.get_span(inds3d)
            final_scores = np.take(scores, inds)
            prediction = [{'span': spans[i], 'score': final_scores[i]} for
                          i in range(len(spans))]
            f.write(json.dumps(prediction) + '\n')
