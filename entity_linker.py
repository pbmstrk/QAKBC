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

from tqdm import tqdm
from transformers import BertTokenizer

EntityPrediction = namedtuple("Prediction", ['kb_id'])


def find_matches(span, text):
    
    matches = []
    index = 0
    while index < len(text):
        index = text.find(span, index)
        matches.append((index, index + len(span)))
        if index == -1:
            break
        index += len(span)
    return matches

def initialise(args):

    linker = EntityLinker(args.model_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    db = DocDB(args.dbpath)

    return linker, tokenizer, db

def fetch_text(doc_id, db):
    return db.get_doc_text(doc_id)


def process_pred(pred, linker, tokenizer, db):
    span = pred['span']
    docids = pred['docs']

    entity_list = []

    doctexts = map(partial(fetch_text, db=db), list(docids))
    for text in doctexts:
        text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(text[0]))
        char_inds = find_matches(span, text) 

        match_dict = linker(text)
        entities = [match_dict.get(ind, -1) for ind in char_inds]
        entities = list(filter(lambda x: x != -1, entities))
        entity_list.extend(entities)
    
    if len(entity_list) > 0:
        entity = Counter(entity_list).most_common(1)[0]
        return EntityPrediction(kb_id=entity[0])
    else:
        return -1

def main(
        preds: str = typer.Argument(..., help="Path to file containing predicted spans"), 
        outdir: str = typer.Argument(..., help="Output directory for prediction file"), 
        model_path: str = typer.Argument(..., help="Path to file containing entity linking model"), 
        dbpath: str = typer.Argument(..., help="Path to SQLite database"),
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
    linker, tokenizer, db = initialise(args)
    logger.info('Finished initialisation')

    # define processing function
    process = partial(process_pred, linker=linker, tokenizer=tokenizer, db=db)

    # define output filename
    outfilename = os.path.splitext(os.path.basename(preds))[0] + "-entity.preds"
    outfilepath = outdir + "/" + outfilename

    # predict entities
    with open(outfilepath, 'w') as outfile:
        with open(preds, 'r') as pred_file:
            for line in tqdm(pred_file):
                lst_of_results = ast.literal_eval(line)
                entities = []
                for result in lst_of_results:
                    ent = process(result)
                    if ent != -1:
                        entities.append(ent.kb_id)
                prediction = {'entities': entities}
                json.dump(prediction, outfile)
                outfile.write("\n")

if __name__ == '__main__':

    typer.run(main)
                



