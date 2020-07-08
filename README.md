# ODQA (Open-domain question answering)

The code in this repository is based on the [DrQA](https://github.com/facebookresearch/DrQA) implementation (see the LICENSE file in the repository). In particular, the retrieval and tokenizer modules are used with slight modifications (adding a BM25 retriever).

Approach to open-domain question answering is divided in two step:
- retrieval: using either TF-IDF or BM25
- reading: using BERT

The code only supports predictions, however, any question answering model that is supported by [huggingface](https://github.com/huggingface/transformers) can be used for the machine comprehension component. 

Prediction can either be completed in one step using `predict.py` or in two stages using `predict_docs.py` for retrieval and `read_docs.py` for reading.

## Retrieval

```
python predict_docs.py dataset outdir ranker retrieverpath --ndocs --logfile
```

`predict_docs.py` creates a `.preds` file  with the questions and the top `ndocs` referenced by their ID in the SQLite database.

## Reading

```
python read_docs.py docs outdir readerpath dbpath --topn --logfile
```

`topn` controls how many predictions are made for each query.
