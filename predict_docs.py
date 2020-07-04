import argparse
import json
import os
import logging

from tqdm import tqdm

from odqa.logger import set_logger
from odqa.retriever import TfidfDocRanker, BM25DocRanker

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('outdir', type=str)
parser.add_argument('ranker', type=str)
parser.add_argument('retrieverpath', type=str)
parser.add_argument('--ndocs', type=int, default=30)
parser.add_argument('--logfile', type=str, default='predict_docs.log')


def get_class(name):
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'bm25':
        return BM25DocRanker


def initialise(args):

    logger.info("Initialising retriever")
    retriever = get_class(args.ranker)(args.retrieverpath)
    return retriever


if __name__ == '__main__':

    # parse command line arguments
    args = parser.parse_args()

    # set up logging
    logger = set_logger(args.logfile)

    # initialise retriever
    retriever = initialise(args)

    # open the dataset of queries and append to list
    queries = []
    for line in open(args.dataset):
        data = json.loads(line)
        queries.append(data['question'])

    # get file name to save predictions to
    basename = os.path.splitext(os.path.basename(args.dataset))[0]
    outfile = os.path.join(args.outdir, basename + '.pdocs')
    logger.info("Saving to {}".format(outfile))

    # retrieve predictions in batches
    with open(outfile, 'w') as f:
        batches = [queries[i: i + 1000] for i in range(0, len(queries), 1000)]
        j = 0
        for i, batch in enumerate(batches):
            # remove this line later
            logger.info('Retrieving results')
            results = retriever.batch_closest_docs(batch, k=args.ndocs)
            logger.info('Writing')
            for result in tqdm(results):
                doc_ids = result[0]
                doc_scores = result[1].tolist()
                d = {'query': queries[j], 'doc_ids': doc_ids, 'doc_scores': doc_scores}
                f.write(json.dumps(d) + '\n')
                j += 1
