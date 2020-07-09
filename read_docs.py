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

from odqa.logger import set_logger
from odqa.reader import BatchReader
from odqa.retriever import DocDB

from transformers import BertModel, BertTokenizer
from torch.nn.functional import softmax

parser = argparse.ArgumentParser()
parser.add_argument('docs', type=str)
parser.add_argument('outdir', type=str)
parser.add_argument('readerpath', type=str)
parser.add_argument('dbpath', type=str)
parser.add_argument('--topn', type=int, default=10)
parser.add_argument('--logfile', type=str, default='read_docs.log')


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
    docscores = np.array(docscores)

    batch = (query, docids, doctexts, docscores)

    return batch

def normalize(arr, axis=0):

    return arr/arr.sum(axis, keepdims=True)


def initialise(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reader = BatchReader(args.readerpath, device)

    ctx = BertModel.from_pretrained('context/').to(device)
    q = BertModel.from_pretrained('question/').to(device)
    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    

    db = DocDB(args.dbpath)

    logger.info('Finished initialisation')

    return reader, db, q, ctx, tok


def fetch_text(doc_id, db):
    return db.get_doc_text(doc_id)


if __name__ == '__main__':

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # parse command line arguments
    args = parser.parse_args()

    # set up logger
    logger = set_logger(args.logfile)

    # initialise reader and DocDB
    reader, db, q, ctx, tok  = initialise(args)

    basename = os.path.splitext(os.path.basename(args.docs))[0]
    outfile = os.path.join(args.outdir, basename + '-' +
                           os.path.basename(args.readerpath) + '.preds')

    logger.info('Output file: {}'.format(outfile))

    logger.info("Retrieving data")

    all_doc_ids = []
    all_doc_scores = []
    queries = []

    for line in open(args.docs):
        dat = json.loads(line)
        all_doc_ids.append(dat['doc_ids'])
        all_doc_scores.append(dat['doc_scores'])
        queries.append(dat['query'])

    logger.info("Reading..")

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
            query, docids, doc_texts, doc_scores = batch
            
            q.eval()
            ctx.eval()
            with torch.no_grad():
                q_encoding = tok(query, return_tensors="pt")
                for key in q_encoding:
                    q_encoding[key] = q_encoding[key].to(device)
                q_val = q(**q_encoding)[1]
                ctx_encoding = tok(doc_texts, return_tensors="pt", padding=True, truncation=True)
                for key in ctx_encoding:
                    ctx_encoding[key] = ctx_encoding[key].to(device)
                ctx_val = ctx(**ctx_encoding)[1]
                scores_dpr = torch.einsum('ij,kj -> k', q_val, ctx_val).detach().cpu().numpy()
                scores_dpr = scores_dpr.clip(min=0)
                scores_dpr = normalize(scores_dpr)

            doc_scores = normalize(doc_scores)

            score_inds = np.argsort(-(0.7*doc_scores + 0.3*scores_dpr))[:30]
            doc_texts = np.array(doc_texts)[score_inds].tolist()
            
            span_scores = reader.predict((query, doc_texts))
            scores = span_scores
            # inds = np.argpartition(scores, -args.topn, axis=None)[-args.topn:]
            # inds = inds[np.argsort(np.take(scores, inds))][::-1]
            # inds3d = zip(*np.unravel_index(inds, scores.shape))
            idx = scores.reshape(scores.shape[0], -1).argmax(-1)
            inds = list((np.arange(scores.shape[0]), *np.unravel_index(idx,
                         scores.shape[-2:])))
            inds = np.ravel_multi_index(inds, dims=scores.shape)
            inds = inds[np.argsort(np.take(scores, inds))][::-1][:args.topn]
            inds3d = list(zip(*np.unravel_index(inds, scores.shape)))
            spans = reader.get_span(inds3d)
            final_scores = np.take(scores, inds)
            prediction = [{'span': spans[i],'document': docids[inds3d[i][0]] , 'score': final_scores[i]} for
                          i in range(len(spans))]
            f.write(json.dumps(prediction) + '\n')
        logger.info("Finished predicting")
