import os
import unittest

from classification_with_embeddings.__main__ import main
from test.test_utils import get_relative_path


class TestMain(unittest.TestCase):
    def test_get_entity_embeddings(self):
        embeddings_out_path = 'word2vec_model.tsv'

        argv = [
            __file__,
            'get-entity-embeddings',
            '--method', 'word2vec',
            '--train-data-path', get_relative_path(__file__, '../mock_data/train.txt'),
            '--output-dir', get_relative_path(__file__, '.'),
        ]
        main(argv)
        self.assertTrue(os.path.exists(get_relative_path(__file__, 'word2vec_model.tsv')))
        os.remove(get_relative_path(__file__, embeddings_out_path))

    def test_train_test_split(self):
        train_out_path = 'data_10_rows_train.txt'
        test_out_path = 'data_10_rows_test.txt'
        argv = [
            __file__,
            'train-test-split',
            '--data-path', get_relative_path(__file__, '../mock_data/data_10_rows.txt'),
            '--output-dir', get_relative_path(__file__, '.'),
        ]
        main(argv)
        self.assertTrue(os.path.exists(get_relative_path(__file__, train_out_path)))
        self.assertTrue(os.path.exists(get_relative_path(__file__, test_out_path)))
        os.remove(get_relative_path(__file__, train_out_path))
        os.remove(get_relative_path(__file__, test_out_path))

    def test_evaluate(self):
        model_out_path = 'word2vec_model.tsv'
        cr_out_path = 'word2vec_cr.txt'
        cm_out_path = 'word2vec_cm.png'
        roc_out_path = 'word2vec_roc.png'

        argv_embeddings = [
            __file__,
            'get-entity-embeddings',
            '--method', 'word2vec',
            '--train-data-path', get_relative_path(__file__, '../mock_data/train.txt'),
            '--output-dir', get_relative_path(__file__, '.'),
        ]
        main(argv_embeddings)

        # using stored embeddings
        argv_evaluate = [
            __file__,
            'evaluate_embeddings_model',
            '--method', 'word2vec',
            '--train-data-path', get_relative_path(__file__, '../mock_data/sanity_check_dataset_train.txt'),
            '--test-data-path', get_relative_path(__file__, '../mock_data/sanity_check_dataset.txt'),
            '--embeddings-path', get_relative_path(__file__, 'word2vec_model.tsv'),
            '--results-path', get_relative_path(__file__, '.'),
        ]
        main(argv_evaluate)
        self._assert_and_remove_output(model_out_path, cr_out_path, cm_out_path, roc_out_path)

        # using grid-search
        argv_evaluate_gs = [
            __file__,
            'evaluate_embeddings_model',
            '--method', 'fasttext',
            '--param-grid-path', get_relative_path(__file__, '../mock_data/param_grid.json'),
            '--cv', '3',
            '--train-data-path', get_relative_path(__file__, '../mock_data/sanity_check_dataset_train.txt'),
            '--test-data-path', get_relative_path(__file__, '../mock_data/sanity_check_dataset_test.txt'),
            '--results-path', get_relative_path(__file__, '.'),
        ]
        main(argv_evaluate_gs)
        self._assert_and_remove_output(model_out_path, cr_out_path, cm_out_path, roc_out_path)

        # using grid-search with multiple documents relating to a single entity
        argv_evaluate_gs = [
            __file__,
            'evaluate_embeddings_model',
            '--method', 'word2vec', 'fasttext',
            '--param-grid-path', get_relative_path(__file__, '../mock_data/param_grid.json'),
            '--cv', '3',
            '--train-data-path', get_relative_path(__file__, '../mock_data/sanity_check_dataset_p1_train.txt'), get_relative_path(__file__, '../mock_data/sanity_check_dataset_p2_train.txt'),
            '--test-data-path', get_relative_path(__file__, '../mock_data/sanity_check_dataset_p1_test.txt'), get_relative_path(__file__, '../mock_data/sanity_check_dataset_p2_test.txt'),
            '--results-path', get_relative_path(__file__, '.'),
        ]
        main(argv_evaluate_gs)
        self._assert_and_remove_output(model_out_path, cr_out_path, cm_out_path, roc_out_path)

    def _assert_and_remove_output(self, model_out_path: str, cr_out_path: str, cm_out_path: str, roc_out_path: str):
        self.assertTrue(os.path.exists(get_relative_path(__file__, model_out_path)))
        self.assertTrue(os.path.exists(get_relative_path(__file__, cr_out_path)))
        self.assertTrue(os.path.exists(get_relative_path(__file__, cm_out_path)))
        self.assertTrue(os.path.exists(get_relative_path(__file__, roc_out_path)))
        os.remove(get_relative_path(__file__, model_out_path))
        os.remove(get_relative_path(__file__, cr_out_path))
        os.remove(get_relative_path(__file__, cm_out_path))
        os.remove(get_relative_path(__file__, roc_out_path))
