from abc import ABC, abstractmethod
from typing import List

import numpy as np
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from sklearn.base import BaseEstimator, TransformerMixin

from classification_with_embeddings.embedding.embed_util import get_aggregate_embeddings


class ADocEmbedder(ABC, BaseEstimator, TransformerMixin):
    """Base class for transformers that perform embedding of documents/sentences and produce matrix-form embeddings."""

    @abstractmethod
    def get_word_to_embedding(self, train_sentences: List[List[str]]):
        pass

    def __init__(self, vector_size: int = 100):
        self.vector_size = vector_size
        self._word_to_embedding = None

    def fit(self, train_sentences: List[List[str]], y):
        self._word_to_embedding = self.get_word_to_embedding(train_sentences)
        return self

    def transform(self, test_sentences: List[List[str]]):
        return get_aggregate_embeddings(test_sentences, self._word_to_embedding, method='average')

    @staticmethod
    def factory(method='word2vec'):
        if method == 'word2vec':
            return Word2VecDocEmbedder()
        if method == 'fasttext':
            return FastTextDocEmbedder()
        elif method == 'doc2vec':
            return Doc2VecDocEmbedder()
        else:
            raise ValueError('Method \'{}\' not implemented.'.format(method))


class Word2VecDocEmbedder(ADocEmbedder):
    def get_word_to_embedding(self, train_sentences):
        word2vec_model = Word2Vec(train_sentences, vector_size=self.vector_size, min_count=1)
        return {k: word2vec_model.wv[k] for k in word2vec_model.wv.index_to_key}


class FastTextDocEmbedder(ADocEmbedder):
    def get_word_to_embedding(self, train_sentences: List[List[str]]):
        ft_model = FastText(vector_size=self.vector_size)
        ft_model.build_vocab(train_sentences)
        ft_model.train(corpus_iterable=train_sentences, total_examples=len(train_sentences), epochs=10)
        return {k: ft_model.wv[k] for k in ft_model.wv.index_to_key}


class Doc2VecDocEmbedder(ADocEmbedder):
    def get_word_to_embedding(self, train_sentences: List[List[str]]):
        raise NotImplementedError('Method not supported.')

    def __init__(self, vector_size: int = 100):
        super().__init__(vector_size=vector_size)
        self.model = None

    def fit(self, train_sentences: List[List[str]], y):
        tagged_data = [TaggedDocument(words=s, tags=[str(i)]) for i, s in enumerate(train_sentences)]
        self.model = Doc2Vec(tagged_data, vector_size=self.vector_size)
        return self

    def transform(self, test_sentences: List[List[str]]):
        return np.vstack([self.model.infer_vector(words) for words in test_sentences])
