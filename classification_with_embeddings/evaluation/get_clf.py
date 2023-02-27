from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from classification_with_embeddings import LABEL_WORD_PREFIX
from classification_with_embeddings.embedding.embed import get_aggregate_embedding
from classification_with_embeddings.evaluation import logger
from classification_with_embeddings.util.arguments import process_param_spec


class Classifier:
    def __init__(self, clf, word_to_embedding):
        self._clf = clf
        self._word_to_embedding = word_to_embedding

    def predict(self, sample):
        aggregate_embedding = get_aggregate_embedding(sample, self._word_to_embedding)
        return str(self._clf.predict(aggregate_embedding.reshape(1, -1))[0])

    def supports_predict_proba(self):
        return hasattr(self._clf, 'predict_proba')

    def predict_proba(self, sample):
        if not hasattr(self._clf, 'predict_proba'):
            raise ValueError()
        aggregate_embedding = get_aggregate_embedding(sample, self._word_to_embedding)
        return self._clf.predict_proba(aggregate_embedding.reshape(1, -1))[0]

    def classes(self):
        return self._clf.classes_


def get_clf_with_internal_clf(word_to_embedding: dict, training_data_path: str, clf_internal=None, internal_clf_args: str = '') -> Classifier:
    """Get internal classifier based classifier.

    :param word_to_embedding: mapping of words to their embeddings
    :param training_data_path: Path to file containing the training data in fastText format
    :param clf_internal: internal classifier to use (use scikit-learn's RandomForestClassifier if None)
    :param internal_clf_args: additional arguments passed to the internal classifier
    :return: function that takes a sample (document in fastText format) and predicts its label
    """

    # train internal classifier
    embeddings = []
    target = []

    logger.info('Obtaining classifier')
    logger.info('Computing training data for internal classifier')
    with open(training_data_path, 'r') as f:
        for idx, t_sample in enumerate(f):
            embeddings.append(get_aggregate_embedding(t_sample, word_to_embedding))
            label_search = [el for el in t_sample.split() if LABEL_WORD_PREFIX in el]
            if len(label_search) == 0:
                raise ValueError('Label not found in training sample {0} in {1}'.format(idx, training_data_path))
            gt_label = label_search[0].replace(LABEL_WORD_PREFIX, '')
            target.append(gt_label)
    x_train = np.vstack(embeddings)

    # get additional parameters and train
    logger.info('Training internal classifier')
    clf_internal_params = process_param_spec(internal_clf_args)
    if clf_internal is None:
        clf_internal = RandomForestClassifier(**clf_internal_params).fit(x_train, target)
    else:
        clf_internal = clf_internal(**clf_internal_params).fit(x_train, target)

    return Classifier(clf_internal, word_to_embedding)


# TODO should return Classifier instance. Implement support for predict_proba
def get_clf_starspace(word_to_embedding: dict) -> Callable[[str], str]:
    """Get StarSpace-based classifier.

    :param word_to_embedding: mapping of words to their embeddings
    :return: function that takes a sample (document in fastText format) and predicts its label
    """

    logger.info('Obtaining classifier')
    label_embeddings = [(key, word_to_embedding[key]) for key in word_to_embedding.keys() if LABEL_WORD_PREFIX in key]
    index_to_label_key = [e[0] for e in label_embeddings]
    label_emb_mat = np.vstack([e[1] for e in label_embeddings])

    def classify(sample: str):
        aggregate_embedding = get_aggregate_embedding(sample, word_to_embedding)
        sims = np.matmul(label_emb_mat, aggregate_embedding)
        return index_to_label_key[np.argmax(sims)].replace(LABEL_WORD_PREFIX, '')

    return classify
