from typing import List

import numpy as np

from classification_with_embeddings import LABEL_WORD_PREFIX
from classification_with_embeddings.embedding.embed_util import get_aggregate_embeddings
from classification_with_embeddings.evaluation.clf.a_classifier import AClassifier


class StarSpaceClassifier(AClassifier):
    """Classifier initialized with a mapping from words and labels to their embeddings obtained using StarSpace."""

    def __init__(self, word_to_embedding: dict):
        self.word_to_embedding = word_to_embedding
        label_embeddings = [(key, word_to_embedding[key]) for key in word_to_embedding.keys() if LABEL_WORD_PREFIX in key]
        self.index_to_label_key = [e[0] for e in label_embeddings]
        self.label_emb_mat = np.transpose([e[1] for e in label_embeddings])

    def predict(self, sentences: List[List[str]]):
        sims = self._get_sims(sentences)
        return [self.index_to_label_key[idx].replace(LABEL_WORD_PREFIX, '') for idx in np.argmax(sims, axis=1)]

    def supports_predict_proba(self):
        return True

    def predict_proba(self, sentences: List[List[str]]):
        sims = self._get_sims(sentences)
        return np.exp(sims) / np.sum(np.exp(sims))

    def classes(self):
        return [e.replace(LABEL_WORD_PREFIX, '') for e in self.index_to_label_key]

    def _get_sims(self, sentences: List[List[str]]):
        aggregate_embeddings = get_aggregate_embeddings(sentences, self.word_to_embedding)
        return np.matmul(aggregate_embeddings, self.label_emb_mat)
