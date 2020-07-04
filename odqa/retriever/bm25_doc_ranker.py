#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Rank documents with TF-IDF scores"""

import logging
import numpy as np
import scipy.sparse as sp

from multiprocessing.pool import ThreadPool
from functools import partial

from . import utils
from .. import tokenizers

logger = logging.getLogger(__name__)


def get_bm25_matrix(cnts, Nt, dls, k1, b):
    """Convert the word count matrix into tfidf one.
    cnts: csr_matrix with shape [hash_size, nb_docs]
    score(D, Q) = idf * (tf * (k1 + 1) / tf + k1 * (1 - b + b * dl / avgdl))
    idf = log((N - Nt + 0.5) / (Nt + 0.5))
    * tf:  term frequency in document
    * N:  number of documents
    * Nt: number of occurrences of term in all documents, shape [hash_size]
    * dls: document lengths, shape [nb_docs]
    * avgdl: average document length
    * k1, b: BM25 parameters
    """
    N = cnts.shape[1]
    idfs = np.log((N - Nt + 0.5) / (Nt + 0.5))
    idfs[idfs < 0] = 0
    idfs = sp.diags(idfs, 0)  # sparse shape [hash_size, hash_size]

    avgdl = np.mean(dls)
    B = (1. - b + b * dls / avgdl)  # [nb_docs]
    B = np.expand_dims(B, 0)  # [1, nb_docs]

    nom = cnts * (k1 + 1.)
    binary = (cnts > 0).astype(np.int32)  # to avoid dense matrix in the following line
    denom = cnts.astype(np.float32)+np.float32(k1) * binary.multiply(B.astype(np.float32))
    tfs = nom.astype(np.float32).multiply(denom.power(-1))  # sparse shape [hash_size, nb_docs]

    bm25_matrix = idfs.dot(tfs)  # sparse shape [hash_size, nb_docs]
    return bm25_matrix


class BM25DocRanker(object):
    """Loads a pre-weighted inverted index of token/document terms.
    Scores new queries by taking sparse dot products.
    """

    def __init__(self, count_path, strict=True):
        """
        Args:
            count_path: path to saved count matrix file
            k1, b: parameters of BM25
            strict: fail on empty queries or continue (and return empty result)
        """
        # Load from disk
        logger.info('Loading %s' % count_path)
        self.doc_mat, metadata = utils.load_sparse_csr(count_path)
        logger.info('Finished loading')

        # Other metadata
        self.ngrams = metadata['ngram']
        self.hash_size = metadata['hash_size']
        self.tokenizer = tokenizers.get_class(metadata['tokenizer'])()
        self.doc_freqs = metadata['doc_freqs']
        self.doc_dict = metadata['doc_dict']
        self.num_docs = len(self.doc_dict[0])
        self.strict = strict

    def get_doc_index(self, doc_id):
        """Convert doc_id --> doc_index"""
        return self.doc_dict[0][doc_id]

    def get_doc_id(self, doc_index):
        """Convert doc_index --> doc_id"""
        return self.doc_dict[1][doc_index]

    def closest_docs(self, query, k=1):
        """Closest docs by dot product between query and documents
        in tfidf weighted word vector space.
        """
        spvec = self.text2spvec(query)
        res = spvec * self.doc_mat

        if len(res.data) <= k:
            o_sort = np.argsort(-res.data)
        else:
            o = np.argpartition(-res.data, k)[0:k]
            o_sort = o[np.argsort(-res.data[o])]

        doc_scores = res.data[o_sort]
        doc_ids = [self.get_doc_id(i) for i in res.indices[o_sort]]
        return doc_ids, doc_scores

    def batch_closest_docs(self, queries, k=1, num_workers=None):
        """Process a batch of closest_docs requests multithreaded.
        Note: we can use plain threads here as scipy is outside of the GIL.
        """
        closest_docs = partial(self.closest_docs, k=k)
        with ThreadPool(num_workers) as threads:
            results = threads.map(closest_docs, queries)

        if len(results) != len(queries):
            logger.warning("Length of results does not match with queries.")

        return results

    def parse(self, query):
        """Parse the query into tokens (either ngrams or tokens)."""
        tokens = self.tokenizer.tokenize(query)
        return tokens.ngrams(n=self.ngrams, uncased=True,
                             filter_fn=utils.filter_ngram)

    def text2spvec(self, query):
        """Create a sparse tfidf-weighted word vector from query.
        tfidf = log(tf + 1) * log((N - Nt + 0.5) / (Nt + 0.5))
        """
        # Get hashed ngrams
        words = self.parse(utils.normalize(query))
        wids = [utils.hash(w, self.hash_size) for w in words]

        if len(wids) == 0:
            if self.strict:
                raise RuntimeError('No valid word in: %s' % query)
            else:
                logger.warning('No valid word in: %s' % query)
                return sp.csr_matrix((1, self.hash_size))

        # Count TF
        wids_unique, wids_counts = np.unique(wids, return_counts=True)
        data = wids_counts  # values are just word counts
        indptr = np.array([0, len(wids_unique)])  # One row, sparse csr matrix
        spvec = sp.csr_matrix((data, wids_unique, indptr), shape=(1, self.hash_size))

        return spvec
