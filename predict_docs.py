from odqa.pipeline.odqa import *
import argparse
from tqdm import tqdm
import json 
import os

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('outdir', type=str)
parser.add_argument('ranker', type=str)
parser.add_argument('retrieverpath', type=str)
parser.add_argument('--ndocs', type=int, default=30)

def get_class(name):
    if name=='tfidf':
        return TfidfDocRanker
    if name=='bm25':
        return BM25DocRanker

def initialise(args):

    retriever = get_class(args.ranker)(args.retrieverpath)
    return retriever

if __name__ == '__main__':

    args = parser.parse_args()
    retriever = initialise(args)

    queries = []
    for line in open(args.dataset):
        data = json.loads(line)
        queries.append(data['question'])

    basename = os.path.splitext(os.path.basename(args.dataset))[0]
    outfile = os.path.join(args.outdir, basename  + '.pdocs')

    with open(outfile, 'w') as f:
            batches = [queries[i: i + 1000] for i in range(0, len(queries), 1000)]
            j = 0
            for i, batch in enumerate(batches):
                # remove this line later
                logging.info('Retrieving results')
                results  = retriever.batch_closest_docs(batch, k=args.ndocs)
                logging.info('Writing')
                for result  in tqdm(results):
                    doc_ids = result[0]
                    doc_scores = result[1].tolist()
                    d = {'query': queries[j], 'doc_ids': doc_ids, 'doc_scores': doc_scores}
                    f.write(json.dumps(d) + '\n')
                    j += 1
