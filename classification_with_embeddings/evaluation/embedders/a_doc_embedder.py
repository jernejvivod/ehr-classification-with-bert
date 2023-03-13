from abc import ABC, abstractmethod
from typing import List

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
    def factory(method: str | List[str] = 'word2vec'):
        if isinstance(method, str):
            if method == 'word2vec':
                from classification_with_embeddings.evaluation.embedders.word2vec_doc_embedder import Word2VecDocEmbedder
                return Word2VecDocEmbedder()
            if method == 'fasttext':
                from classification_with_embeddings.evaluation.embedders.fasttext_doc_embedder import FastTextDocEmbedder
                return FastTextDocEmbedder()
            elif method == 'doc2vec':
                from classification_with_embeddings.evaluation.embedders.doc2vec_doc_embedder import Doc2VecDocEmbedder
                return Doc2VecDocEmbedder()
            else:
                raise ValueError('Method \'{}\' not implemented.'.format(method))
        elif isinstance(method, List):
            from classification_with_embeddings.evaluation.embedders.composite_embedder import CompositeEmbedder
            return CompositeEmbedder(embedders=[ADocEmbedder.factory(m) for m in method])
        else:
            raise ValueError('Argument \'method\' must be specified as either str or list.')
