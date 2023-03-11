from typing import Dict, List

import numpy as np

from classification_with_embeddings.embedding.embed_util import get_aggregate_embeddings
from classification_with_embeddings.evaluation.clf.a_classifier import AClassifier


class Classifier(AClassifier):
    """Basic classifier initialized with a Scikit-Learn's classifier for internal use and
    a mapping from words to their embeddings.
    """

    def __init__(self, clf, word_to_embedding: Dict[str, np.ndarray[1, ...]]):
        self._clf = clf
        self._word_to_embedding = word_to_embedding

    def predict(self, sentences: List[List[str]]) -> np.ndarray[1, ...]:
        aggregate_embeddings = get_aggregate_embeddings(sentences, self._word_to_embedding)
        return self._clf.predict(aggregate_embeddings)

    def supports_predict_proba(self):
        return hasattr(self._clf, 'predict_proba')

    def predict_proba(self, sentences: List[List[str]]):
        if not hasattr(self._clf, 'predict_proba'):
            raise ValueError()
        aggregate_embedding = get_aggregate_embeddings(sentences, self._word_to_embedding)
        return self._clf.predict_proba(aggregate_embedding)

    def classes(self):
        return self._clf.classes_
