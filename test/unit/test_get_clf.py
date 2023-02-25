import unittest

from classification_with_embeddings.embedding.embed import get_word_to_embedding
from classification_with_embeddings.evaluation.get_clf import get_clf_with_internal_clf
from test.test_utils import _get_relative_path


class TestGetClf(unittest.TestCase):
    def test_get_clf_with_internal_clf(self):
        word_to_embedding = get_word_to_embedding(_get_relative_path(__file__, '../mock_data/mock_model.tsv'))
        training_data_path = _get_relative_path(__file__, '../mock_data/train.txt')
        clf = get_clf_with_internal_clf(word_to_embedding, training_data_path)
        self.assertIsNotNone(clf)
        self.assertIn(clf("this is a simple test"), ['0', '1'])
        self.assertIn(clf("terminal altitude"), ['0', '1'])

    # TODO implement
    def get_clf_starspace(self):
        pass
