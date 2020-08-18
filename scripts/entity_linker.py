import ast
import re
import os
import json

import argparse
from collections import namedtuple
from collections import Counter, OrderedDict

from qakgc.retriever import DocDB
from qakgc.logger import set_logger
from qakgc.linker import EntityLinker
from functools import partial


from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer


def initialise(args, logger):

    linker = EntityLinker(args.model_path, logger)
    tokenizer = BertWordPieceTokenizer('bert-base-uncased-vocab.txt', lowercase=True)
    db = DocDB(args.db_path)
    return linker, tokenizer, db

def fetch_text(doc_id, db):
    return db.get_doc_text(doc_id)


def white_space_fix(text):
    return re.sub(r"\s+", "", text) 


def process_result_list(res_list, linker, tokenizer, db, index_map, logger=None):

    # query is the same for each example
    query = res_list[0]['query']
    data_to_link = []

    for pred in res_list:
        docids = pred['docs']
        span = pred['span']
        start_idx = pred['start_idx']
        end_idx = pred['end_idx']

        idx = list(zip(start_idx, end_idx))

        doctexts = map(partial(fetch_text, db=db), list(docids))
        
        for i, text in enumerate(doctexts):
            ids = idx[i]
            text = text[0]
            encoding = tokenizer.encode(query, text)
            start = encoding.token_to_chars(ids[0])[0]
            #if ids[1] >= len(encoding.tokens):
            #    end = encoding.token_to_chars(len(encoding.tokens)-2)[1]
            #else:
            end = encoding.token_to_chars(ids[1])[1]
            d = {
                "id": 0,
                "label": "unknown",
                "label_id": -1,
                "context_left": text[:start].lower(),
                "mention": text[start:end].lower(),
                "context_right": text[end:].lower()
            }
            data_to_link.append(d)
 
            if white_space_fix(span) != white_space_fix(d['mention']):
                logger.info("Mismatch! Span: {} \t Match: {}".format(white_space_fix(span), white_space_fix(d['mention'])))

    predictions = linker(data_to_link)
    predictions = process_predictions(predictions, index_map)
    return predictions

def process_predictions(predictions, index_map):

    results = []
    for lst in predictions:
        for el in lst:
            if index_map[str(el)] not in results:
                results.append(index_map[str(el)])
                break

    return results


def main(args):

    # set up logger
    logger = set_logger(args.log_file)
    logger.info('Arguments: %s' % str(args))

    # initialise reader and DocDB
    linker, tokenizer, db = initialise(args, logger)
    logger.info('Finished initialisation')

    # load index map file
    with open(args.index_map_path) as json_file:
        index_map = json.load(json_file)

    # define processing function
    process = partial(process_result_list, linker=linker, tokenizer=tokenizer, db=db, 
        index_map=index_map, logger=logger)

    # define output filename
    outfilename = os.path.splitext(os.path.basename(args.preds))[0] + "-entity.preds"
    outfilepath = args.output_dir + "/" + outfilename

    # predict entities
    with open(outfilepath, 'w') as outfile:
        with open(args.preds, 'r') as pred_file:
            for line in tqdm(pred_file):
                lst_of_results = ast.literal_eval(line)
                entities = process(lst_of_results)
                prediction = {'entities': entities}
                json.dump(prediction, outfile)
                outfile.write("\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--preds",
        default=None,
        type=str,
        required=True,
        metavar='\b',
        help="Path to .preds file containing span predicitons",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        metavar='\b',
        help="The output directory where predictions will be written",
    )

    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=True,
        metavar='\b',
        help="Path to trained model",
    )

    parser.add_argument(
        "--db_path",
        default=None,
        type=str,
        required=True,
        metavar='\b',
        help="Path to SQLite DB",
    )

    parser.add_argument(
        "--index_map_path",
        default=None,
        type=str,
        required=True,
        metavar='\b',
        help="Path to file mapping index ids to entities",
    )

    parser.add_argument(
        "--log_file",
        default="entity_linker.log",
        type=str,
        metavar='\b',
        help="Path to log file (default: %(default)s)",
    )

    args = parser.parse_args()

    main(args)

                



