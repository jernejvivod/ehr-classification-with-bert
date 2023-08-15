from typing import List, Iterator, Union, Optional

import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from classification_with_embeddings.evaluation.embedders.a_doc_embedder import ADocEmbedder


class Doc2VecDocEmbedder(ADocEmbedder):
    def __init__(self, embedding_kwargs: Optional[dict] = None, **model_init_kwargs):
        super().__init__(embedding_kwargs=embedding_kwargs, **model_init_kwargs)
        self.model = None

    def get_word_to_embedding(self, train_sentences: Union[List[List[str]], Iterator], y: list):
        raise NotImplementedError('Method not supported.')

    def fit(self, X: Union[List[List[str]], Iterator], y: list):
        tagged_data = [TaggedDocument(words=s, tags=[str(i)]) for i, s in enumerate(X)]
        self.model = Doc2Vec(tagged_data, **self.embedding_kwargs)
        return self

    def transform(self, X: List[List[str]]):
        return np.vstack([self.model.infer_vector(words) for words in X])
