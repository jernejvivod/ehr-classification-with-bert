from typing import List

from classification_with_embeddings import LABEL_WORD_PREFIX
from classification_with_embeddings.evaluation.clf.a_classifier import AClassifier


class PipelineClassifier(AClassifier):
    """Classifier initialized with a classifier (pipeline) that performs embedding and predicting
    as part of the prediction operation.
    """

    def __init__(self, clf):
        self._clf = clf

    def predict(self, samples: List[List[str]]):
        return self._clf.predict([[w for w in sample if LABEL_WORD_PREFIX not in w] for sample in samples])

    def supports_predict_proba(self):
        return hasattr(self._clf, 'predict_proba')

    def predict_proba(self, samples: List[List[str]]):
        if not hasattr(self._clf, 'predict_proba'):
            raise ValueError('Classifier does not support predict_proba.')
        return self._clf.predict_proba([[w for w in sample if LABEL_WORD_PREFIX not in w] for sample in samples])

    def classes(self):
        return self._clf.classes_
