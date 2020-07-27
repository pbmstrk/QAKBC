import argparse
import json
from odqa.logger import set_logger

def hits_at_k(preds, ans, k):
    """accepts list of predictions and answers"""

    preds_k = preds[:k]

    any_in = lambda a, b: bool(set(a).intersection(b))

    return any_in(preds_k, ans)


def evaluate(dataset_file, prediction_file, k):

    answers = []
    for line in open(dataset_file):
        data = json.loads(line)
        answer = data['entity']
        answers.append(answer)
    
    predictions = []
    with open(prediction_file) as f:
        for line in f:
            data = json.loads(line)
            preds = data['entities']
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
    parser.add_argument('--map_file', type=str)

    # parse command line arguments
    args = parser.parse_args()

    # set up logger
    logger = set_logger(args.logfile)

    logger.info(25*"-" + " Evaluation " + 25*"-")
    
    logger.info('Arguments: %s' % str(args))

    evaluate(args.datasetfile, args.predfile, args.k)