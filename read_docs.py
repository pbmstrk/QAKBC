import typer
import json
import os
import torch

from functools import partial
from tqdm import tqdm
from torch.utils import data
from collections import namedtuple

from odqa.logger import set_logger
from odqa.reader import Reader
from odqa.retriever import DocDB
from odqa.reader import get_predictions

from transformers import BertModel, BertTokenizer



class ReaderDataset(data.Dataset):

    def __init__(self, queries, docids, docscores):
        self.queries = queries
        self.docids = docids
        self.docscores = docscores

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        X = self.queries[idx]
        Y = self.docids[idx]
        Z = self.docscores[idx]
        return X, Y, Z

def fetch_text(doc_id, db):
    return db.get_doc_text(doc_id)


def generate_batch(batch, db, tokenizer):

    query = batch[0][0]
    docids = list(set(batch[0][1]))

    doctexts = map(partial(fetch_text, db=db), docids)
    doctexts = [text[0] for text in doctexts]
    
    raw_inputs = [(query, doctext) for doctext in doctexts]

    encoding = tokenizer(raw_inputs, padding=True,
                                  truncation="only_second",
                                  return_tensors="pt")

    inputs = {
        'input_ids': encoding['input_ids'].unsqueeze(0),
        'attention_mask': encoding['attention_mask'].unsqueeze(0)
    }

    return inputs, docids

def initialise(args):

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoder = BertModel.from_pretrained('bert-base-uncased')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reader = Reader(encoder, 768).to(device)

    reader.load_state_dict(torch.load(args.checkpointfile, map_location=device))

    db = DocDB(args.dbpath)

    return tokenizer, reader, db, device

def main(
        docs: str = typer.Argument(..., help="Path to file containing predicted documents"), 
        outdir: str = typer.Argument(..., help="Output directory for prediction file"), 
        checkpointfile: str = typer.Argument(..., help="Path to file containing model checkpoint"), 
        dbpath: str = typer.Argument(..., help="Path to SQLite database"),
        logfile: str = typer.Option('read_docs.log', help="Path to log file")
    ):

    # set up args datastructure
    args_dict = locals()
    ArgsClass = namedtuple('args', sorted(args_dict))
    args = ArgsClass(**args_dict)

    # set up logger
    logger = set_logger(logfile)

    # initialise models
    tokenizer, reader, db, device = initialise(args)
    logger.info('Finished initialisation')

    # set output name
    basename = os.path.splitext(os.path.basename(docs))[0]
    outfile = os.path.join(outdir, basename + '-' +
                           os.path.splitext(os.path.basename(checkpointfile))[0] + '.preds')
    logger.info('Output file: {}'.format(outfile))

    # read in data
    logger.info("Retrieving data")
    all_doc_ids = []
    all_doc_scores = []
    queries = []

    for line in open(docs):
        dat = json.loads(line)
        all_doc_ids.append(dat['doc_ids'])
        all_doc_scores.append(dat['doc_scores'])
        queries.append(dat['query'])
    
    # read documents
    logger.info("Reading..")
    collate_fn = partial(generate_batch, db=db, tokenizer=tokenizer)

    querydataset = ReaderDataset(queries, all_doc_ids, all_doc_scores)
    data_generator = data.DataLoader(querydataset, batch_size=1,
                                     collate_fn=collate_fn, num_workers=0)

    with open(outfile, 'w') as f:

        for batch, docids in tqdm(data_generator, total=len(queries)):

            for key in batch.keys():
                batch[key] = batch[key].to(device)

            with torch.no_grad():
                model_outputs = reader(**batch)

            predictions = get_predictions(batch, model_outputs, tokenizer)[0]
            preds = [{'span': pred.text, 'score': pred.prob, 'docs': [docids[int(passage)] for passage in pred.passage_idx], 
                'start_idx': [int(idx) for idx in pred.start_idx], 'end_idx': [int(idx) for idx in pred.end_idx]} for pred in predictions]

            f.write(json.dumps(preds) + '\n')
                
        logger.info("Finished predicting")

if __name__ == '__main__':

    typer.run(main)