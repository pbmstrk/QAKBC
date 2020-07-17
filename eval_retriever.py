import argparse
import json
import os
import string
import re

from tqdm import tqdm

from odqa.logger import set_logger
from odqa.retriever import TfidfDocRanker, BM25DocRanker, DocDB

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('ranker', type=str)
parser.add_argument('retrieverpath', type=str)
parser.add_argument('dbpath', type=str)
parser.add_argument('--ndocs', type=int, default=30)
parser.add_argument('--logfile', type=str, default='eval_retriever.log')

def get_class(name):
    if name == 'tfidf':
        return TfidfDocRanker
    if name == 'bm25':
        return BM25DocRanker

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def initialise(args):

    logger.info("Initialising retriever")
    retriever = get_class(args.ranker)(args.retrieverpath)
    db = DocDB(args.dbpath)

    return retriever, db

def fetch_text(doc_id, db):
    return db.get_doc_text(doc_id)

if __name__ == '__main__':

    # parse command line arguments
    args = parser.parse_args()

    # set up logging
    logger = set_logger(args.logfile)

    # initialise retriever
    retriever, db = initialise(args)

    # open the dataset of queries and append to list
    queries = []
    answers = []
    for line in open(args.dataset):
        data = json.loads(line)
        queries.append(data['question'])
        answers.append(data['answer'])

    batches = [queries[i: i + 1000] for i in range(0, len(queries), 1000)]
    j = 0
    count = 0
    total = 0
    for i, batch in enumerate(batches):
        logger.info('Retrieving results')
        results = retriever.batch_closest_docs(batch, k=args.ndocs)
        for result in tqdm(results):
            doc_ids = result[0]
            answer = answer[j]
            total += 1
            for doc_id in doc_ids:
                text = normalize_answer(fetch_text(doc_id, db))
                for ans in answer:
                    if normalize_answer(ans) in text:
                        count += 1
                        break
            j += 1
        logger.info(str(count/total))
