import unittest

import numpy as np
from gensim.models import Doc2Vec

from classification_with_embeddings.evaluation import get_clf
from classification_with_embeddings.embedding.embed_util import get_word_to_embedding
from classification_with_embeddings.evaluation.get_clf import get_clf_with_internal_clf, get_clf_starspace, get_clf_with_internal_clf_doc2vec, get_clf_with_internal_clf_gs
from test.test_utils import get_relative_path


class TestGetClf(unittest.TestCase):
    test_sample1 = [['this', 'is', 'a', 'simple', 'test']]
    test_sample2 = [['terminal', 'altitude']]
    test_sample3 = [['this', 'is', 'a', 'simple', 'test'], ['terminal', 'altitude']]
    test_samples = [test_sample1, test_sample2, test_sample3]

    test_sample_multiple1 = [[['this', 'is', 'a', 'simple', 'test'], ['terminal', 'altitude']]]
    test_sample_multiple2 = [[['this', 'is', 'a', 'simple', 'test'], ['this', 'must', 'be', 'a', 'test']], [['terminal', 'altitude'], ['never', 'surreneder']]]
    test_samples_multiple = [test_sample_multiple1, test_sample_multiple2]

    def test_get_clf_with_internal_clf(self):
        word_to_embedding = get_word_to_embedding(get_relative_path(__file__, '../mock_data/mock_model.tsv'))
        training_data_path = get_relative_path(__file__, '../mock_data/train.txt')
        clf = get_clf_with_internal_clf(word_to_embedding, training_data_path)
        self._assert_pred(clf, self.test_samples)

    def test_get_clf_with_internal_clf_gs(self):
        clf = get_clf_with_internal_clf_gs(
            get_relative_path(__file__, '../mock_data/data_10_rows.txt'),
            get_relative_path(__file__, '../mock_data/data_10_rows.txt'),
            param_grid={},
            embedding_method='word2vec'
        )
        self._assert_pred(clf, self.test_samples)

    def test_get_clf_with_internal_clf_gs_multiple(self):
        file_data = get_relative_path(__file__, '../mock_data/data_10_rows.txt')

        clf = get_clf_with_internal_clf_gs(
            [file_data, file_data],
            [file_data, file_data],
            param_grid={},
            embedding_method=['word2vec', 'fasttext']
        )
        self._assert_pred(clf, self.test_samples_multiple)

    def test_get_clf_with_internal_clf_doc2vec(self):
        doc2vec_model = Doc2Vec.load(get_relative_path(__file__, '../mock_data/doc2vec_model.bin'))
        training_data_path = get_relative_path(__file__, '../mock_data/train.txt')
        clf = get_clf_with_internal_clf_doc2vec(doc2vec_model, training_data_path)
        self._assert_pred(clf, self.test_samples)

    def test_get_clf_starspace(self):
        word_to_embedding = get_word_to_embedding(get_relative_path(__file__, '../mock_data/mock_starspace_model.tsv'))
        clf = get_clf_starspace(word_to_embedding)
        self._assert_pred(clf, self.test_samples)

    def test_process_param_grid_for_set_params(self):
        param_grid = {
            'e1__param_1': [1, 2, 3],
            'e1__param_2': 'test',
            'e2__param_3': ['ab', 'cd', 'ef']
        }

        param_grid_expected = {
            'e1__param_1': 1,
            'e1__param_2': 'test',
            'e2__param_3': 'ab'
        }

        param_grid_proc = get_clf._process_param_grid_for_set_params(param_grid)

        self.assertEqual(param_grid_expected, param_grid_proc)

    def _assert_pred(self, clf, test_samples):
        self.assertIsNotNone(clf)

        def assert_probs(probs: np.ndarray):
            for p in probs:
                self.assertEqual(2, len(p))
                for e in p:
                    self.assertTrue(0.0 <= e <= 1.0)

        for sample in test_samples:
            self.assertTrue(all(el in ['0', '1'] for el in clf.predict(sample)))
            probs = clf.predict_proba(sample)
            assert_probs(probs)
