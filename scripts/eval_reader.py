import argparse
import json
import unicodedata
import string
import logging
import re

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Logging to a file
file_handler = logging.FileHandler('eval.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s :%(levelname)s : %(name)s: %(message)s'))
logger.addHandler(file_handler)

# Logging to console
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s :%(levelname)s : %(name)s: %(message)s'))
logger.addHandler(stream_handler)


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

def hits_at_k(preds, ans, k):
    """accepts list of predictions and answers"""

    preds_k = [normalize_answer(pred) for pred in preds[:k]]
    ans = [normalize_answer(a) for a in ans]

    any_in = lambda a, b: bool(set(a).intersection(b))

    return any_in(preds_k, ans)

def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


def evaluate(dataset_file, prediction_file, k):

    answers = []
    for line in open(dataset_file):
        data = json.loads(line)
        answer = [normalize(a) for a in data['answer']]
        answers.append(answer)
    
    predictions = []
    with open(prediction_file) as f:
        for line in f:
            data = json.loads(line)
            preds = [normalize(d['span']) for d in data]
            predictions.append(preds)


    score = 0
    for i in range(len(predictions)):
        score += hits_at_k(predictions[i], answers[i], k)

    total = len(predictions)
    hitsk = 100.0 * score / total
    logger.info({'Hits@{}'.format(k): hitsk})

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('datasetfile',
                    help="path to files that should be combined")
    parser.add_argument('predfile',
                    help="path to outputfile, path/to/output.txt")
    parser.add_argument('k', type=int,
                    help="Hits@k")
    
    logger.info(25*"-" + " Evaluation " + 25*"-")
    args = parser.parse_args()
    logger.info('Arguments: %s' % str(args))
    evaluate(args.datasetfile, args.predfile, args.k)