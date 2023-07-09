from typing import List, Iterator, Union

import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from classification_with_embeddings.evaluation.embedders.a_doc_embedder import ADocEmbedder


class Doc2VecDocEmbedder(ADocEmbedder):
    def __init__(self, vector_size: int = 100, **kwargs):
        super().__init__(vector_size=vector_size, **kwargs)
        self.model = None

    def get_word_to_embedding(self, train_sentences: Union[List[List[str]], Iterator], y: list):
        raise NotImplementedError('Method not supported.')

    def fit(self, X: Union[List[List[str]], Iterator], y: list):
        tagged_data = [TaggedDocument(words=s, tags=[str(i)]) for i, s in enumerate(X)]
        self.model = Doc2Vec(tagged_data, vector_size=self.vector_size, **self.method_kwargs)
        return self

    def transform(self, X: List[List[str]]):
        return np.vstack([self.model.infer_vector(words) for words in X])
