from odqa.pipeline.odqa import *
import argparse
from tqdm import tqdm
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('outdir', type=str)
parser.add_argument('readermodel', type=str)
parser.add_argument('retrievermodel', type=str)
parser.add_argument('docdb', type=str)
parser.add_argument('--ndocs', type=int, default=10)
parser.add_argument('--topn', type=int, default=1)

if __name__ == '__main__':
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ODQA(
        reader_model = args.readermodel,
        retriever_path = args.retrievermodel,
        doc_db_path = args.docdb,
        device = device
    )

    queries = []
    for line in open(args.dataset):
        data = json.loads(line)
        queries.append(data['question'])
    
    basename = os.path.splitext(os.path.basename(args.dataset))[0]
    outfile = os.path.join(args.outdir, basename + '-' + args.readermodel + '.preds')

    with open(outfile, 'w') as f:
        for query in tqdm(queries):
            prediction = model.process_query(query, ndocs=args.ndocs, topn=args.topn)
            f.write(json.dumps(prediction) + '\n')
            


