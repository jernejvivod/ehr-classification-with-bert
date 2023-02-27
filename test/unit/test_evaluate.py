import os
import unittest

from classification_with_embeddings.embedding.embed import get_word_to_embedding
from classification_with_embeddings.evaluation.evaluate import evaluate
from classification_with_embeddings.evaluation.get_clf import get_clf_with_internal_clf
from test.test_utils import get_relative_path


class TestEvaluate(unittest.TestCase):

    def test_evaluate(self):
        method_name = 'word2vec'
        cm_suffix = '.png'
        cr_suffix = '_cr.txt'

        word_to_embedding = get_word_to_embedding(get_relative_path(__file__, '../mock_data/mock_model.tsv'))
        training_data_path = get_relative_path(__file__, '../mock_data/train.txt')
        clf = get_clf_with_internal_clf(word_to_embedding, training_data_path)
        evaluate(clf, method_name, get_relative_path(__file__, '../mock_data/test.txt'), '')

        self.assertTrue(os.path.exists(method_name + cm_suffix))
        self.assertTrue(os.path.exists(method_name + cr_suffix))
        os.remove(method_name + cm_suffix)
        os.remove(method_name + cr_suffix)
