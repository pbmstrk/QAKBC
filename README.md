# ODQA (Open-domain question answering)

The code in this repository is based on the [DrQA](https://github.com/facebookresearch/DrQA) implementation (see the LICENSE file in the repository). In particular, the retrieval and tokenizer modules are used with slight modifications (adding a BM25 retriever).

Approach to open-domain question answering is divided in two step:
- retrieval: using either TF-IDF or BM25
- reading: using BERT

The code only supports predictions, however, any question answering model that is supported by [huggingface](https://github.com/huggingface/transformers) can be used for the machine comprehension component. 

Prediction can either be completed in one step using `predict.py` or in two stages using `predict_docs.py` for retrieval and `read_docs.py` for reading.

## Retrieval

```
Usage: predict_docs.py [OPTIONS] DATASET OUTDIR RANKER RETRIEVERPATH

Arguments:
  DATASET        Path to file containing queries  [required]
  OUTDIR         Output directory for prediction file  [required]
  RANKER         Ranker to use  [required]
  RETRIEVERPATH  Path to retriever  [required]

Options:
  --ndocs INTEGER                 Number of documents to retrieve  [default:
                                  30]

  --logfile TEXT                  Path to log file  [default:
                                  predict_docs.log]
```

`predict_docs.py` creates a `.preds` file  with the questions and the top `ndocs` referenced by their ID in the SQLite database.

## Reading

```
Usage: read_docs.py [OPTIONS] DOCS OUTDIR CHECKPOINTFILE DBPATH

Arguments:
  DOCS            Path to file containing predicted documents  [required]
  OUTDIR          Output directory for prediction file  [required]
  CHECKPOINTFILE  Path to file containing model checkpoint  [required]
  DBPATH          Path to SQLite database  [required]

Options:
  --logfile TEXT                  Path to log file  [default: read_docs.log]
```
## Entity Linking

```
Usage: entity_linker.py [OPTIONS] PREDS OUTDIR MODEL_PATH DBPATH

Arguments:
  PREDS       Path to file containing predicted spans  [required]
  OUTDIR      Output directory for prediction file  [required]
  MODEL_PATH  Path to file containing entity linking model  [required]
  DBPATH      Path to SQLite database  [required]

Options:
  --logfile TEXT                  Path to log file  [default:
                                  entity_linker.log]
```

