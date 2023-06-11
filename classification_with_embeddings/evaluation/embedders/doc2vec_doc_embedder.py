from typing import List, Iterator, Union

import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from classification_with_embeddings.evaluation.embedders.a_doc_embedder import ADocEmbedder


class Doc2VecDocEmbedder(ADocEmbedder):
    def get_word_to_embedding(self, train_sentences: Union[List[List[str]], Iterator]):
        raise NotImplementedError('Method not supported.')

    def __init__(self, vector_size: int = 100):
        super().__init__(vector_size=vector_size)
        self.model = None

    def fit(self, train_sentences: Union[List[List[str]], Iterator], y):
        tagged_data = [TaggedDocument(words=s, tags=[str(i)]) for i, s in enumerate(train_sentences)]
        self.model = Doc2Vec(tagged_data, vector_size=self.vector_size, **self.method_kwargs)
        return self

    def transform(self, test_sentences: List[List[str]]):
        return np.vstack([self.model.infer_vector(words) for words in test_sentences])
