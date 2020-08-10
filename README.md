# QAKGC: Open-Domain Question Answering for Knowledge Graph Completion

Approach to open-domain question answering for knowledge graph completion is divided in three steps:
- retrieval: either TF-IDF or BM25
- reading: BERT
- entity linking: BLINK

## Retrieval

```
usage: predict_docs.py [-h] --data --output_dir --ranker {bm25,tfidf} --retriever_path --db_path [--n_docs] [--log_file]

optional arguments:
  -h, --help            show this help message and exit
  --data              Path to file containing queries
  --output_dir        The output directory where predictions will be written
  --ranker {bm25,tfidf}
                        Ranker to use for document retrieval
  --retriever_path    Path to retriever model
  --db_path           Path to SQLite DB
  --n_docs            Number of documents to retrieve (default: 30)
  --log_file          Path to log file (default: predict_docs.log)
```

`predict_docs.py` creates a `.preds` file  with the questions and the top `ndocs` referenced by their ID in the SQLite database.

## Reading

```
usage: read_docs.py [-h] --docs --output_dir --model_path --db_path [--log_file]

optional arguments:
  -h, --help      show this help message and exit
  --docs        File containing predicted documents
  --output_dir  The output directory where predictions will be written
  --model_path  Path to trained model
  --db_path     Path to SQLite DB
  --log_file    Path to log file (default: read_docs.log)
```
## Entity Linking

```
usage: entity_linker.py [-h] --preds --output_dir --model_path --db_path --index_map_path [--log_file]

optional arguments:
  -h, --help          show this help message and exit
  --preds           Path to .preds file containing span predicitons
  --output_dir      The output directory where predictions will be written
  --model_path      Path to trained model
  --db_path         Path to SQLite DB
  --index_map_path  Path to file mapping index ids to entities
  --log_file        Path to log file (default: entity_linker.log)
```


