import ast
import os
import json

import typer
from collections import namedtuple
from collections import Counter

from odqa.retriever import DocDB
from odqa.logger import set_logger
from odqa.linker import EntityLinker
from functools import partial

from collections import OrderedDict

from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer


def initialise(args, logger):

    linker = EntityLinker(args.model_path, logger)
    tokenizer = BertWordPieceTokenizer('bert-base-uncased-vocab.txt', lowercase=True)
    db = DocDB(args.dbpath)
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


def main(
        preds: str = typer.Argument(..., help="Path to file containing predicted spans"), 
        outdir: str = typer.Argument(..., help="Output directory for prediction file"), 
        model_path: str = typer.Argument(..., help="Path to file containing entity linking model"), 
        dbpath: str = typer.Argument(..., help="Path to SQLite database of documents"),
        index_map_path: str = typer.Argument(..., help="Path to file mapping index ids to entities"),
        logfile: str = typer.Option('entity_linker.log', help="Path to log file")
    ):

    # set up args datastructure
    args_dict = locals()
    ArgsClass = namedtuple('args', sorted(args_dict))
    args = ArgsClass(**args_dict)

    # set up logger
    logger = set_logger(logfile)
    logger.info(args)

    # initialise reader and DocDB
    linker, tokenizer, db = initialise(args, logger)
    logger.info('Finished initialisation')

    # load index map file
    with open(index_map_path) as json_file:
        index_map = json.load(json_file)

    # define processing function
    process = partial(process_result_list, linker=linker, tokenizer=tokenizer, db=db, 
        index_map=index_map)

    # define output filename
    outfilename = os.path.splitext(os.path.basename(preds))[0] + "-entity.preds"
    outfilepath = outdir + "/" + outfilename

    # predict entities
    with open(outfilepath, 'w') as outfile:
        with open(preds, 'r') as pred_file:
            for line in tqdm(pred_file):
                lst_of_results = ast.literal_eval(line)
                entities = process(lst_of_results)
                prediction = {'entities': entities}
                json.dump(prediction, outfile)
                outfile.write("\n")

if __name__ == '__main__':

    typer.run(main)
                



