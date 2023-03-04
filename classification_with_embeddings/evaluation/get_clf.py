from abc import ABC, abstractmethod
from typing import Callable, Dict

import numpy as np
from gensim.models import Doc2Vec
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from classification_with_embeddings import LABEL_WORD_PREFIX
from classification_with_embeddings.embedding.embed_util import get_aggregate_embedding
from classification_with_embeddings.evaluation import logger
from classification_with_embeddings.util.arguments import process_param_spec


class AClassifier(ABC):
    @abstractmethod
    def predict(self, sample):
        pass

    @abstractmethod
    def predict_proba(self, sample) -> np.ndarray[1, ...]:
        pass

    @abstractmethod
    def supports_predict_proba(self) -> bool:
        pass

    @abstractmethod
    def classes(self) -> list:
        pass


class Classifier(AClassifier):
    def __init__(self, clf, word_to_embedding: Dict[str, np.ndarray[1, ...]]):
        self._clf = clf
        self._word_to_embedding = word_to_embedding

    def predict(self, sample: str):
        aggregate_embedding = get_aggregate_embedding(sample, self._word_to_embedding)
        return str(self._clf.predict(aggregate_embedding.reshape(1, -1))[0])

    def supports_predict_proba(self):
        return hasattr(self._clf, 'predict_proba')

    def predict_proba(self, sample: str):
        if not hasattr(self._clf, 'predict_proba'):
            raise ValueError()
        aggregate_embedding = get_aggregate_embedding(sample, self._word_to_embedding)
        return self._clf.predict_proba(aggregate_embedding.reshape(1, -1))[0]

    def classes(self):
        return self._clf.classes_


class Doc2VecClassifier(AClassifier):
    def __init__(self, clf, doc2vec_model: Doc2Vec):
        self._clf = clf
        self._doc2vec_model = doc2vec_model

    def predict(self, sample: str):
        words = [w for w in sample.split() if LABEL_WORD_PREFIX not in w]
        aggregate_embedding = self._doc2vec_model.infer_vector(words)
        return str(self._clf.predict(aggregate_embedding.reshape(1, -1))[0])

    def supports_predict_proba(self):
        return hasattr(self._clf, 'predict_proba')

    def predict_proba(self, sample: str):
        if not hasattr(self._clf, 'predict_proba'):
            raise ValueError()
        words = [w for w in sample.split() if LABEL_WORD_PREFIX not in w]
        aggregate_embedding = self._doc2vec_model.infer_vector(words)
        return self._clf.predict_proba(aggregate_embedding.reshape(1, -1))[0]

    def classes(self):
        return self._clf.classes_


class StarSpaceClassifier(AClassifier):
    def __init__(self, word_to_embedding: dict):
        self.word_to_embedding = word_to_embedding
        label_embeddings = [(key, word_to_embedding[key]) for key in word_to_embedding.keys() if LABEL_WORD_PREFIX in key]
        self.index_to_label_key = [e[0] for e in label_embeddings]
        self.label_emb_mat = np.vstack([e[1] for e in label_embeddings])

    def predict(self, sample: str):
        sims = self._get_sims(sample)
        return self.index_to_label_key[np.argmax(sims)].replace(LABEL_WORD_PREFIX, '')

    def supports_predict_proba(self):
        return True

    def predict_proba(self, sample: str):
        sims = self._get_sims(sample)
        return np.exp(sims) / np.sum(np.exp(sims))

    def classes(self):
        return [e.replace(LABEL_WORD_PREFIX, '') for e in self.index_to_label_key]

    def _get_sims(self, sample: str):
        aggregate_embedding = get_aggregate_embedding(sample, self.word_to_embedding)
        return np.matmul(self.label_emb_mat, aggregate_embedding)


def get_clf_with_internal_clf(word_to_embedding: dict[str, np.ndarray[1, ...]], training_data_path: str, clf_internal: ClassifierMixin = None, internal_clf_args: str = '') -> AClassifier:
    """Get internal classifier based classifier.

    :param word_to_embedding: mapping of words to their embeddings
    :param training_data_path: Path to file containing the training data in fastText format
    :param clf_internal: internal classifier to use (use scikit-learn's RandomForestClassifier if None)
    :param internal_clf_args: additional arguments passed to the internal classifier
    :return: initialized AClassifier instance
    """

    def get_aggr_embedding(sample: str):
        return get_aggregate_embedding(sample, word_to_embedding)

    def get_clf(clf):
        return Classifier(clf, word_to_embedding)

    return init_clf(get_aggr_embedding, get_clf, training_data_path, clf_internal, internal_clf_args)


def get_clf_with_internal_clf_doc2vec(doc2vec_model: Doc2Vec, training_data_path: str, clf_internal: ClassifierMixin = None, internal_clf_args: str = ''):
    """Get internal classifier based classifier for Doc2Vec model.

    :param doc2vec_model: gensim's Doc2Vec model
    :param training_data_path: Path to file containing the training data in fastText format
    :param clf_internal: internal classifier to use (use scikit-learn's RandomForestClassifier if None)
    :param internal_clf_args: additional arguments passed to the internal classifier
    :return: initialized AClassifier instance
    """

    def get_aggr_embedding(sample: str):
        words = [w for w in sample.split() if LABEL_WORD_PREFIX not in w]
        return doc2vec_model.infer_vector(words)

    def get_clf(clf):
        return Doc2VecClassifier(clf, doc2vec_model)

    return init_clf(get_aggr_embedding, get_clf, training_data_path, clf_internal, internal_clf_args)


def get_clf_starspace(word_to_embedding: dict) -> AClassifier:
    """Get StarSpace-based classifier.

    :param word_to_embedding: mapping of words to their embeddings
    :return: initialized AClassifier instance
    """

    logger.info('Obtaining classifier.')
    return StarSpaceClassifier(word_to_embedding)


def init_clf(get_aggr_embedding: Callable[[str], np.ndarray[1, ...]], get_clf: Callable[[ClassifierMixin], AClassifier], training_data_path: str, clf_internal=None, internal_clf_args: str = ''):
    """Initialize AClassifier instance.

    :param get_aggr_embedding: function mapping a string sample (document) to an aggregate embedding
    :param get_clf: function mapping a trained internal classifier to an AClassifier instance
    :param training_data_path: Path to file containing the training data in fastText format
    :param clf_internal: internal classifier to use (use scikit-learn's RandomForestClassifier if None)
    :param internal_clf_args: additional arguments passed to the internal classifier
    :return: initialized AClassifier instance
    """
    # train internal classifier
    embeddings = []
    target = []

    logger.info('Obtaining classifier.')
    logger.info('Computing training data for internal classifier.')
    with open(training_data_path, 'r') as f:
        for idx, t_sample in enumerate(f):
            embeddings.append(get_aggr_embedding(t_sample))
            label_search = [el for el in t_sample.split() if LABEL_WORD_PREFIX in el]
            if len(label_search) == 0:
                raise ValueError('Label not found in training sample {0} in {1}'.format(idx, training_data_path))
            gt_label = label_search[0].replace(LABEL_WORD_PREFIX, '')
            target.append(gt_label)
    x_train = np.vstack(embeddings)

    # get additional parameters and train
    logger.info('Training internal classifier.')
    clf_internal_params = process_param_spec(internal_clf_args)
    if clf_internal is None:
        clf_internal = RandomForestClassifier(**clf_internal_params).fit(x_train, target)
    else:
        clf_internal = clf_internal(**clf_internal_params).fit(x_train, target)

    return get_clf(clf_internal)
