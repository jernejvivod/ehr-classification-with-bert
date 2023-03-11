import unittest

import numpy as np
from gensim.models import Doc2Vec

from classification_with_embeddings.embedding.embed_util import get_word_to_embedding
from classification_with_embeddings.evaluation.get_clf import get_clf_with_internal_clf, get_clf_starspace, get_clf_with_internal_clf_doc2vec, get_clf_with_internal_clf_gs
from test.test_utils import get_relative_path


class TestGetClf(unittest.TestCase):
    test_sample1 = [['this', 'is', 'a', 'simple', 'test']]
    test_sample2 = [['terminal', 'altitude']]
    test_sample3 = [['this', 'is', 'a', 'simple', 'test'], ['terminal', 'altitude']]

    def test_get_clf_with_internal_clf(self):
        word_to_embedding = get_word_to_embedding(get_relative_path(__file__, '../mock_data/mock_model.tsv'))
        training_data_path = get_relative_path(__file__, '../mock_data/train.txt')
        clf = get_clf_with_internal_clf(word_to_embedding, training_data_path)
        self._assert_pred(clf)

    def test_get_clf_with_internal_clf_gs(self):
        clf = get_clf_with_internal_clf_gs(get_relative_path(__file__, '../mock_data/data_10_rows.txt'), get_relative_path(__file__, '../mock_data/data_10_rows.txt'), param_grid={}, embedding_method='word2vec')
        self._assert_pred(clf)

    def test_get_clf_with_internal_clf_doc2vec(self):
        doc2vec_model = Doc2Vec.load(get_relative_path(__file__, '../mock_data/doc2vec_model.bin'))
        training_data_path = get_relative_path(__file__, '../mock_data/train.txt')
        clf = get_clf_with_internal_clf_doc2vec(doc2vec_model, training_data_path)
        self._assert_pred(clf)

    def test_get_clf_starspace(self):
        word_to_embedding = get_word_to_embedding(get_relative_path(__file__, '../mock_data/mock_starspace_model.tsv'))
        clf = get_clf_starspace(word_to_embedding)
        self._assert_pred(clf)

    def _assert_pred(self, clf):
        self.assertIsNotNone(clf)

        self.assertTrue(all(el in ['0', '1'] for el in clf.predict(self.test_sample1)))
        self.assertTrue(all(el in ['0', '1'] for el in clf.predict(self.test_sample2)))
        self.assertTrue(all(el in ['0', '1'] for el in clf.predict(self.test_sample3)))

        def assert_probs(probs: np.ndarray[..., 2]):
            for r in probs:
                self.assertEqual(2, len(r))
                for e in r:
                    self.assertTrue(0.0 <= e <= 1.0)

        prob1 = clf.predict_proba(self.test_sample1)
        prob2 = clf.predict_proba(self.test_sample2)
        prob3 = clf.predict_proba(self.test_sample3)

        for prob in (prob1, prob2, prob3):
            assert_probs(prob)
