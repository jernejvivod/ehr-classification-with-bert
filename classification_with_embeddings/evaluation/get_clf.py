from typing import Callable, List, Iterable, Union, Dict

import numpy as np
from gensim.models import Doc2Vec
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from classification_with_embeddings import LABEL_WORD_PREFIX
from classification_with_embeddings.embedding.embed_util import get_aggregate_embedding
from classification_with_embeddings.evaluation import logger
from classification_with_embeddings.evaluation.clf.a_classifier import AClassifier
from classification_with_embeddings.evaluation.clf.classifier import Classifier
from classification_with_embeddings.evaluation.clf.doc2vec_classifier import Doc2VecClassifier
from classification_with_embeddings.evaluation.clf.pipeline_classifier import PipelineClassifier
from classification_with_embeddings.evaluation.clf.starspace_classifier import StarSpaceClassifier
from classification_with_embeddings.evaluation.embedders.a_doc_embedder import ADocEmbedder
from classification_with_embeddings.evaluation.util import _fasttext_data_to_x_y, _fasttext_data_to_x_y_multiple
from classification_with_embeddings.util.arguments import process_param_spec


def get_clf_with_internal_clf(word_to_embedding: Dict[str, np.ndarray],
                              training_data_path: str,
                              clf_internal=None,
                              internal_clf_args: str = '') -> AClassifier:
    """Get internal classifier-based classifier using on stored embeddings.

    :param word_to_embedding: mapping of words to their embeddings
    :param training_data_path: Path to file containing the training data in fastText format
    :param clf_internal: internal classifier to use (use scikit-learn's RandomForestClassifier if None)
    :param internal_clf_args: additional arguments passed to the internal classifier
    :return: initialized Classifier instance
    """

    def get_aggr_embedding(sample: List[str]):
        return get_aggregate_embedding(sample, word_to_embedding)

    def get_clf(clf):
        return Classifier(clf, word_to_embedding)

    return _init_clf(get_aggr_embedding, get_clf, training_data_path, clf_internal, internal_clf_args)


def get_clf_with_internal_clf_gs(train_data_path: Union[str, Iterable[str]],
                                 validation_data_path: Union[str,  Iterable[str]],
                                 param_grid: dict,
                                 embedding_method: Union[str, List[str]] = 'word2vec',
                                 clf_internal=RandomForestClassifier,
                                 cv: int = 5) -> AClassifier:
    """get internal classifier-based classifier (used within pipeline) with parameters tuned using grid-search.

    :param train_data_path: path to file containing the training data in fastText format
    :param validation_data_path: path to file containing the validation data in fastText format
    :param param_grid: parameter grid to use
    :param embedding_method: embedding method to use ('word2vec', 'fasttext', 'doc2vec', or 'starspace')
    :param clf_internal: internal classifier to use
    :param cv: number of folds to use when doing cross-validation
    :return: initialized PipelineClassifier instance
    """

    # define pipeline
    clf_pipeline = Pipeline([('embedding', ADocEmbedder.factory(embedding_method)), ('scaling', RobustScaler()), ('classification', clf_internal())])

    # run grid search
    grid_search = GridSearchCV(
        estimator=clf_pipeline,
        param_grid=param_grid if param_grid is not None else dict(),
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )

    validation_sentences, validation_labels = _fasttext_data_to_x_y(validation_data_path) if \
        isinstance(validation_data_path, str) else _fasttext_data_to_x_y_multiple(validation_data_path)

    logger.info('Performing grid-search.')
    grid_search.fit(validation_sentences, validation_labels)

    # train best estimator
    train_sentences, train_labels = _fasttext_data_to_x_y(train_data_path) if \
        isinstance(train_data_path, str) else _fasttext_data_to_x_y_multiple(train_data_path)
    logger.info('Training best model.')
    clf = grid_search.best_estimator_.fit(train_sentences, train_labels)

    # return PipelineClassifier instance initialized with best estimator
    return PipelineClassifier(clf)


def get_clf_with_internal_clf_doc2vec(doc2vec_model: Doc2Vec,
                                      training_data_path: str,
                                      clf_internal=None,
                                      internal_clf_args: str = ''):
    """Get internal classifier-based classifier using a stored Doc2Vec model.

    :param doc2vec_model: gensim's Doc2Vec model
    :param training_data_path: path to file containing the training data in fastText format
    :param clf_internal: internal classifier to use (use scikit-learn's RandomForestClassifier if None)
    :param internal_clf_args: additional arguments passed to the internal classifier
    :return: initialized Doc2VecClassifier instance
    """

    def get_aggr_embedding(sample: List[str]):
        words = [w for w in sample if LABEL_WORD_PREFIX not in w]
        return doc2vec_model.infer_vector(words)

    def get_clf(clf):
        return Doc2VecClassifier(clf, doc2vec_model)

    return _init_clf(get_aggr_embedding, get_clf, training_data_path, clf_internal, internal_clf_args)


def get_clf_starspace(word_to_embedding: dict) -> AClassifier:
    """Get StarSpace-based classifier using stored embeddings.

    :param word_to_embedding: mapping of words to their embeddings
    :return: initialized StarSpaceClassifier instance
    """

    logger.info('Obtaining classifier.')
    return StarSpaceClassifier(word_to_embedding)


def _init_clf(get_aggr_embedding: Callable[[List[str]], np.ndarray],
              get_clf: Callable[[ClassifierMixin], AClassifier],
              training_data_path: str,
              clf_internal=None,
              internal_clf_args: str = ''):
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
            t_sample_split = t_sample.split()
            embeddings.append(get_aggr_embedding(t_sample_split))
            label_search = [el for el in t_sample_split if LABEL_WORD_PREFIX in el]
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
