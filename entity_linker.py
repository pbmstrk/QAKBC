import ast
import os
import json

import argparse
from collections import namedtuple
from collections import Counter, OrderedDict

from odqa.retriever import DocDB
from odqa.logger import set_logger
from odqa.linker import EntityLinker
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


def process_result_list(res_list, linker, tokenizer, db, index_map):

    # query is the same for each example
    query = res_list[0]['query']
    data_to_link = []

    for pred in res_list:
        docids = pred['docs']
        start_idx = pred['start_idx']
        end_idx = pred['end_idx']

        idx = list(zip(start_idx, end_idx))

        doctexts = map(partial(fetch_text, db=db), list(docids))
        
        for i, text in enumerate(doctexts):
            ids = idx[i]
            encoding = tokenizer.encode(query, text[0])
            start = encoding.token_to_chars(ids[0])[0]
            end = encoding.token_to_chars(ids[1])[1]
            d = {
                "id": 0,
                "label": "unknown",
                "label_id": -1,
                "context_left": text[0][:start].lower(),
                "mention": text[0][start:end],
                "context_right": text[0][end:]
            }
            data_to_link.append(d)
 
        
    predictions = linker(data_to_link)
    predictions = [index_map[str(pred[0])] for pred in predictions]
    predictions = list(OrderedDict.fromkeys(predictions))
    return predictions


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
        index_map=index_map)

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
        help="Path to log file",
    )

    args = parser.parse_args()

    main(args)

                



