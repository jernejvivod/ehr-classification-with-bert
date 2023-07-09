from abc import ABC, abstractmethod
from typing import List, Iterator, Union, Dict
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from classification_with_embeddings.embedding.embed_util import get_aggregate_embeddings


class ADocEmbedder(ABC, BaseEstimator, TransformerMixin):
    """Base class for transformers that perform embedding of documents/sentences and produce matrix-form embeddings."""

    @abstractmethod
    def get_word_to_embedding(self, train_sentences: Union[List[List[str]], Iterator], y: list) -> Dict[str, np.ndarray]:
        pass

    def __init__(self, vector_size: int = 100, **kwargs):
        self.vector_size = vector_size
        self.method_kwargs = {}
        self._word_to_embedding = None

    def fit(self, X: Union[List[List[str]], Iterator], y: list):
        self._word_to_embedding = self.get_word_to_embedding(X, y)
        return self

    def transform(self, X: List[List[str]]):
        return get_aggregate_embeddings(X, self._word_to_embedding, method='average')

    def __call__(self, X: List[List[str]]):
        return self.transform(X)

    @staticmethod
    def factory(method: Union[str, List[str]] = 'word2vec', **kwargs):
        """Factory method to obtain ADocEmbedder instance.

        :param method: embedding method to use (which ADocEmbedder subtype to return)
        :param kwargs: additional keyword arguments to pass to the constructor
        """
        if isinstance(method, str):
            if method == 'word2vec':
                from classification_with_embeddings.evaluation.embedders.word2vec_doc_embedder import Word2VecDocEmbedder
                return Word2VecDocEmbedder(**kwargs)
            if method == 'fasttext':
                from classification_with_embeddings.evaluation.embedders.fasttext_doc_embedder import FastTextDocEmbedder
                return FastTextDocEmbedder(**kwargs)
            elif method == 'doc2vec':
                from classification_with_embeddings.evaluation.embedders.doc2vec_doc_embedder import Doc2VecDocEmbedder
                return Doc2VecDocEmbedder(**kwargs)
            elif method == 'pre-trained-from-file':
                from classification_with_embeddings.evaluation.embedders.pretrained_from_file_doc_embedder import PreTrainedFromFileDocEmbedder
                return PreTrainedFromFileDocEmbedder(**kwargs)
            elif method == 'starspace':
                from classification_with_embeddings.evaluation.embedders.starspace_doc_embedder import StarspaceDocEmbedder
                return StarspaceDocEmbedder(**kwargs)
            else:
                raise ValueError('Method \'{}\' not implemented.'.format(method))
        elif isinstance(method, List):
            from classification_with_embeddings.evaluation.embedders.composite_embedder import CompositeEmbedder
            return CompositeEmbedder(embedders=[ADocEmbedder.factory(m) for m in method])
        else:
            raise ValueError('Argument \'method\' must be specified as either str or list.')
