from odqa.pipeline.odqa import *
from functools import partial
import argparse
from tqdm import tqdm
import json
import os
from collections import OrderedDict
parser = argparse.ArgumentParser()
parser.add_argument('docs', type=str)
parser.add_argument('outdir', type=str)
parser.add_argument('readerpath', type=str)
parser.add_argument('dbpath', type=str)
parser.add_argument('--topn', type=int, default=10)
from sklearn.preprocessing import normalize

def initialise(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reader = BatchReader(args.readerpath, device)

    db = DocDB(args.dbpath)

    logging.info('Finished initialisation')

    return reader, db

def fetch_text(doc_id, db):
    return db.get_doc_text(doc_id)

if __name__ == '__main__':

    args = parser.parse_args()
    reader, db = initialise(args)
    
    basename = os.path.splitext(os.path.basename(args.docs))[0]
    outfile = os.path.join(args.outdir, basename + '-' + os.path.basename(args.readerpath) + '.preds')

    logging.info("Retrieving data")

    all_doc_ids = []
    all_doc_scores = []
    queries = []

    for line in open(args.docs):
        data = json.loads(line)
        all_doc_ids.append(data['doc_ids'])
        all_doc_scores.append(data['doc_scores'])
        queries.append(data['query'])

    logging.info("Reading..")

    with open(outfile, 'w') as f:
        for i in tqdm(range(len(all_doc_ids))):
            docids, docscores = all_doc_ids[i], all_doc_scores[i]
            query = queries[i]

            d = OrderedDict(list(zip(docids, docscores)))
            doc_texts = map(partial(fetch_text, db=db), list(d.keys()))
            doc_texts = [text[0] for text in doc_texts]
            doc_scores = normalize(np.array(docscores)[:, np.newaxis],axis=0)[:, np.newaxis]
            
            batch = (query, doc_texts)
            span_scores  = reader.predict(batch, topn=args.topn)
            scores = (1 - 0.7)*doc_scores + 0.7*span_scores
            inds = np.argpartition(scores, -args.topn, axis=None)[-args.topn:]
            inds = inds[np.argsort(np.take(scores, inds))][::-1]
            inds3d = zip(*np.unravel_index(inds, scores.shape))
            spans = reader.get_span(inds3d)
            final_scores = np.take(scores, inds)
            prediction = [{'span': spans[i], 'score': final_scores[i]} for i in range(len(spans))]
            f.write(json.dumps(prediction) + '\n')
            
            
