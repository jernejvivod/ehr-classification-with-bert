from typing import List

import numpy as np
from gensim.models import Doc2Vec

from classification_with_embeddings.evaluation.clf.a_classifier import AClassifier


class Doc2VecClassifier(AClassifier):
    """Classifier initialized with a Scikit-Learn's classifier for internal use
    and an initialized Doc2Vec model.
    """

    def __init__(self, clf, doc2vec_model: Doc2Vec):
        self._clf = clf
        self._doc2vec_model = doc2vec_model

    def predict(self, sentences: List[List[str]]):
        aggregate_embeddings = np.vstack([self._doc2vec_model.infer_vector(sentence) for sentence in sentences])
        return self._clf.predict(aggregate_embeddings)

    def supports_predict_proba(self):
        return hasattr(self._clf, 'predict_proba')

    def predict_proba(self, sentences: List[List[str]]):
        if not hasattr(self._clf, 'predict_proba'):
            raise ValueError('Classifier does not support predicting probabilities.')
        aggregate_embeddings = np.vstack([self._doc2vec_model.infer_vector(sentence) for sentence in sentences])
        return self._clf.predict_proba(aggregate_embeddings)

    def classes(self):
        return self._clf.classes_
