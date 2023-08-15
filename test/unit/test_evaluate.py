import os
import unittest

from classification_with_embeddings.embedding.embed_util import get_word_to_embedding
from classification_with_embeddings.evaluation.evaluate import evaluate_embeddings_model
from classification_with_embeddings.evaluation.get_clf import get_clf_with_internal_clf
from test.test_utils import get_relative_path


class TestEvaluate(unittest.TestCase):

    def test_evaluate(self):
        method_name = 'word2vec'
        cm_suffix = '_cm.png'
        roc_suffix = '_roc.png'
        cr_suffix = '_cr.txt'
        data_suffix = '_data.txt'

        word_to_embedding = get_word_to_embedding(get_relative_path(__file__, '../mock_data/mock_model.tsv'))
        training_data_path = get_relative_path(__file__, '../mock_data/train.txt')
        clf = get_clf_with_internal_clf(word_to_embedding, training_data_path)
        evaluate_embeddings_model(clf, method_name, get_relative_path(__file__, '../mock_data/test.txt'), '')

        self.assertTrue(os.path.exists(method_name + cm_suffix))
        self.assertTrue(os.path.exists(method_name + roc_suffix))
        self.assertTrue(os.path.exists(method_name + cr_suffix))
        self.assertTrue(os.path.exists(method_name + data_suffix))
        os.remove(method_name + cm_suffix)
        os.remove(method_name + roc_suffix)
        os.remove(method_name + cr_suffix)
        os.remove(method_name + data_suffix)
