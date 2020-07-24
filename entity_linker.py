import spacy
import re
import ast
import os
import json

import argparse
from collections import namedtuple
from collections import Counter

from odqa.retriever import DocDB
from odqa.logger import set_logger
from functools import partial

from tqdm import tqdm


EntityPrediction = namedtuple("Prediction", ['kb_id'])


def initialise(args):

    nlp = spacy.load(args.model_path)
    db = DocDB(args.dbpath)

    logger.info('Finished initialisation')

    return nlp, db

def fetch_text(doc_id, db):
    return db.get_doc_text(doc_id)


def process_pred(pred, nlp):
    span = pred['span']
    docids = pred['docs']

    entity_list = []

    doctexts = map(partial(fetch_text, db=db), list(docids))
    for text in doctexts:
        matches = re.finditer(span, text)
        char_inds = [(match.start(), match.end()) for match in matches]

        doc = nlp(text)
        match_dict = {(e.start_char, e.end_char): [e.kb_id_] for e in doc.ents}
        entities = [match_dict.get(ind, -1) for ind in char_inds]
        entities = list(filter(lambda x: x != -1, entities))
        entity_list.extend(entities)

    entity = Counter(entity_list).most_common(1)[0]

    return EntityPrediction(kb_id=entity)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('preds', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('dbpath', type=str)
    parser.add_argument('--logfile', type=str, default='entity_linker.log')

    # parse command line arguments
    args = parser.parse_args()

    # set up logger
    logger = set_logger(args.logfile)

    # initialise reader and DocDB
    nlp, db = initialise(args)

    # define processing function
    process = partial(process_pred, nlp=nlp)

    # define outputpath
    outfilename = os.path.splitext(os.path.basename(args.preds))[0] + "-entity.preds"
    outfilepath = args.outdir + "/" + outfilename

    with open(outfilepath, 'w') as outfile:
        with open(args.preds, 'r') as pred_file:
            for line in tqdm(pred_file):
                lst_of_results = ast.literal_eval(line)
                entities = []
                for result in lst_of_results:
                    ent = process(result)
                    entities.append(ent.kb_id)
                prediction = {'entities': entities}
                json.dump(prediction, outfile)
                outfile.write("\n")

                



