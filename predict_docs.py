import typer
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

    retriever = get_class(args.ranker)(args.retrieverpath)
    return retriever

def main(
        dataset: str = typer.Argument(..., help="Path to file containing queries"), 
        outdir: str = typer.Argument(..., help="Output directory for prediction file"), 
        ranker: str = typer.Argument(..., help="Ranker to use"), 
        retrieverpath: str = typer.Argument(..., help="Path to retriever"),
        ndocs: int = typer.Option(30, help="Number of documents to retrieve"),
        logfile: str = typer.Option('predict_docs.log', help="Path to log file")
    ):

    # set up args datastructure
    args_dict = locals()
    ArgsClass = namedtuple('args', sorted(args_dict))
    args = ArgsClass(**args_dict)

    # set up logging
    logger = set_logger(logfile)

    logger.info(args)

    # initialise retriever
    logger.info("Initialising retriever")
    retriever = initialise(args)

    # open the dataset of queries
    queries = []
    for line in open(dataset):
        data = json.loads(line)
        queries.append(data['question'])

    # get file name to save predictions to
    basename = os.path.splitext(os.path.basename(dataset))[0]
    outfile = os.path.join(outdir, basename + '.pdocs')
    logger.info("Saving to {}".format(outfile))

    # retrieve predictions in batches
    with open(outfile, 'w') as f:
        batches = [queries[i: i + 1000] for i in range(0, len(queries), 1000)]
        j = 0
        for i, batch in enumerate(batches):
            # remove this line later
            logger.info('Retrieving results')
            results = retriever.batch_closest_docs(batch, k=ndocs)
            logger.info('Writing')
            for result in tqdm(results):
                doc_ids = result[0]
                doc_scores = result[1].tolist()
                d = {'query': queries[j], 'doc_ids': doc_ids, 'doc_scores': doc_scores}
                f.write(json.dumps(d) + '\n')
                j += 1
        logger.info("Finished predicting")


if __name__ == '__main__':

    typer.run(main)
