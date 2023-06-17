import os.path
import unittest
from typing import Final

from gensim.models import Doc2Vec
from sklearn.ensemble import RandomForestClassifier

from classification_with_embeddings.embedding.embed import get_word2vec_embeddings, get_fasttext_embeddings, get_starspace_embeddings, get_doc2vec_embeddings
from classification_with_embeddings.embedding.embed_util import get_word_to_embedding
from classification_with_embeddings.evaluation.evaluate import evaluate_embeddings_model
from classification_with_embeddings.evaluation.get_clf import get_clf_starspace, get_clf_with_internal_clf, get_clf_with_internal_clf_doc2vec, get_clf_with_internal_clf_gs
from classification_with_embeddings.train_test_split.train_test_split import get_train_test_split
from test.test_utils import get_relative_path


class TestEmbed(unittest.TestCase):
    TRAINING_SET_PATH: Final = get_relative_path(__file__, '../mock_data/sanity_check_dataset_train.txt')
    TEST_SET_PATH: Final = get_relative_path(__file__, '../mock_data/sanity_check_dataset_test.txt')
    OUT_PATH: Final = get_relative_path(__file__, 'sanity_check_dataset_results/')
    STARSPACE_PATH: Final = get_relative_path(__file__, '../../embedding_methods/StarSpace/starspace')
    BIO_WORD_VEC_PATH: Final = get_relative_path(__file__, '../../embedding_methods/BioWordVec/bio_embedding_extrinsic')

    TRAIN_SET_SUFFIX_FOR_GS: Final = "gs_train"
    VALIDATION_SET_SUFFIX_FOR_GS: Final = "gs_validation"
    TRAINING_SET_PATH_GS: Final = get_relative_path(__file__, './sanity_check_dataset_train_' + TRAIN_SET_SUFFIX_FOR_GS + '.txt')
    VALIDATION_SET_PATH_GS: Final = get_relative_path(__file__, './sanity_check_dataset_train_' + VALIDATION_SET_SUFFIX_FOR_GS + '.txt')

    def test_classification_sanity_check(self):
        self._run_test_classification_sanity_check(embedding_method='word2vec')
        self._run_test_classification_sanity_check(embedding_method='fasttext')
        self._run_test_classification_sanity_check(embedding_method='doc2vec')
        self._run_test_classification_sanity_check(embedding_method='starspace')
        self._run_test_classification_sanity_check(embedding_method='pre-trained-from-file')

    def test_classification_gs_sanity_check(self):
        self._get_train_validation_split(self.TRAINING_SET_PATH, get_relative_path(__file__, '.'))

        self._run_test_classification_gs_sanity_check(embedding_method='word2vec', param_grid=dict())
        self._run_test_classification_gs_sanity_check(embedding_method='fasttext', param_grid=dict())
        self._run_test_classification_gs_sanity_check(embedding_method='doc2vec', param_grid=dict())
        self._run_test_classification_gs_sanity_check(embedding_method='starspace', param_grid=dict())
        self._run_test_classification_gs_sanity_check(embedding_method='pre-trained-from-file', param_grid=dict())

    def _run_test_classification_sanity_check(self, embedding_method: str, args: str = '', clf_internal=None, internal_clf_args: str = ''):
        if embedding_method == 'word2vec':
            get_word2vec_embeddings(self.TRAINING_SET_PATH, self.OUT_PATH, args)
            word_to_embedding = get_word_to_embedding(get_relative_path(__file__, os.path.join(self.OUT_PATH, '{}_model.tsv'.format(embedding_method))))
            clf = get_clf_with_internal_clf(word_to_embedding, self.TRAINING_SET_PATH, clf_internal=clf_internal, internal_clf_args=internal_clf_args)
        elif embedding_method == 'fasttext':
            get_fasttext_embeddings(self.TRAINING_SET_PATH, self.OUT_PATH, args)
            word_to_embedding = get_word_to_embedding(get_relative_path(__file__, os.path.join(self.OUT_PATH, '{}_model.tsv'.format(embedding_method))))
            clf = get_clf_with_internal_clf(word_to_embedding, self.TRAINING_SET_PATH, clf_internal=clf_internal, internal_clf_args=internal_clf_args)
        elif embedding_method == 'doc2vec':
            get_doc2vec_embeddings(self.TRAINING_SET_PATH, self.OUT_PATH, args)
            doc2vec_model = Doc2Vec.load(get_relative_path(__file__, os.path.join(self.OUT_PATH, 'doc2vec_model.bin')))
            clf = get_clf_with_internal_clf_doc2vec(doc2vec_model, self.TRAINING_SET_PATH, clf_internal=clf_internal, internal_clf_args=internal_clf_args)
        elif embedding_method == 'starspace':
            get_starspace_embeddings(self.STARSPACE_PATH, self.TRAINING_SET_PATH, self.OUT_PATH, args)
            word_to_embedding = get_word_to_embedding(get_relative_path(__file__, os.path.join(self.OUT_PATH, '{}_model.tsv'.format(embedding_method))))
            clf = get_clf_starspace(word_to_embedding)
        elif embedding_method == 'pre-trained-from-file':
            word_to_embedding = get_word_to_embedding(self.BIO_WORD_VEC_PATH, binary=True)
            clf = get_clf_with_internal_clf(word_to_embedding, self.TRAINING_SET_PATH, clf_internal=clf_internal, internal_clf_args=internal_clf_args)
        else:
            raise NotImplementedError('Unknown method {}.'.format(embedding_method))

        evaluate_embeddings_model(clf, embedding_method, get_relative_path(__file__, self.TEST_SET_PATH), self.OUT_PATH)

    def _run_test_classification_gs_sanity_check(self, embedding_method: str, param_grid: dict, clf_internal=RandomForestClassifier):
        clf = get_clf_with_internal_clf_gs(self.TRAINING_SET_PATH_GS, self.VALIDATION_SET_PATH_GS, clf_internal=clf_internal, param_grid=param_grid, embedding_method=embedding_method)
        evaluate_embeddings_model(clf, embedding_method, get_relative_path(__file__, self.TEST_SET_PATH), self.OUT_PATH)

    def _get_train_validation_split(self, data_path: str, output_dir: str):
        get_train_test_split(data_path, output_dir, train_size=0.7, train_suffix=self.TRAIN_SET_SUFFIX_FOR_GS, test_suffix=self.VALIDATION_SET_SUFFIX_FOR_GS)
