import argparse
import json
import os

from tqdm import tqdm
from collections import namedtuple

from odqa.logger import set_logger
from odqa.retriever import TfidfDocRanker, BM25DocRanker


def get_class(name):
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'bm25':
        return BM25DocRanker


def initialise(args):

    retriever = get_class(args.ranker)(args.retriever_path)
    return retriever

def main(args):

    # set up logging
    logger = set_logger(args.log_file)

    logger.info(args)

    # initialise retriever
    logger.info("Initialising retriever")
    retriever = initialise(args)

    # open the dataset of queries
    queries = []
    for line in open(args.data):
        data = json.loads(line)
        queries.append(data['question'])

    # get file name to save predictions to
    basename = os.path.splitext(os.path.basename(args.data))[0]
    outfile = os.path.join(args.output_dir, basename + '.pdocs')
    logger.info("Saving to {}".format(outfile))

    # retrieve predictions in batches
    with open(outfile, 'w') as f:
        batches = [queries[i: i + 1000] for i in range(0, len(queries), 1000)]
        j = 0
        for i, batch in enumerate(batches):
            # remove this line later
            logger.info('Retrieving results')
            results = retriever.batch_closest_docs(batch, k=args.n_docs)
            logger.info('Writing')
            for result in tqdm(results):
                doc_ids = result[0]
                doc_scores = result[1].tolist()
                d = {'query': queries[j], 'doc_ids': doc_ids, 'doc_scores': doc_scores}
                f.write(json.dumps(d) + '\n')
                j += 1
        logger.info("Finished predicting")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data",
        default=None,
        type=str,
        required=True,
        help="Path to file containing queries",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where predictions will be written",
    )

    parser.add_argument(
        "--ranker",
        default=None,
        type=str,
        required=True,
        choices=["bm25", "tfidf"],
        help="Ranker to use for document retrieval",
    )

    parser.add_argument(
        "--retriever_path",
        default=None,
        type=str,
        required=True,
        help="Path to retriever model",
    )

    parser.add_argument(
        "--db_path",
        default=None,
        type=str,
        required=True,
        help="Path to SQLite DB",
    )

    parser.add_argument(
        "--n_docs",
        default=30,
        type=int,
        help="Number of documents to retrieve",
    )

    parser.add_argument(
        "--log_file",
        default="predict_docs.log",
        type=str,
        help="Path to log file",
    )

    args = parser.parse_args()

    main(args)

