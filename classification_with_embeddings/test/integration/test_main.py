import os
import unittest

from classification_with_embeddings.__main__ import main
from classification_with_embeddings.test.test_utils import _get_relative_path


class TestMain(unittest.TestCase):
    def test_get_entity_embeddings(self):
        embeddings_out_path = 'word2vec_model.tsv'

        argv = [
            __file__,
            'get-entity-embeddings',
            '--method', 'word2vec',
            '--train-data-path', _get_relative_path(__file__, '../mock_data/train.txt'),
            '--output-dir', _get_relative_path(__file__, '.'),
        ]
        main(argv)
        self.assertTrue(os.path.exists(_get_relative_path(__file__, 'word2vec_model.tsv')))
        os.remove(_get_relative_path(__file__, embeddings_out_path))

    def test_train_test_split(self):
        train_out_path = 'data_10_rows_train.txt'
        test_out_path = 'data_10_rows_test.txt'
        argv = [
            __file__,
            'train-test-split',
            '--data-path', _get_relative_path(__file__, '../mock_data/data_10_rows.txt'),
            '--output-dir', _get_relative_path(__file__, '.'),
        ]
        main(argv)
        self.assertTrue(os.path.exists(_get_relative_path(__file__, train_out_path)))
        self.assertTrue(os.path.exists(_get_relative_path(__file__, test_out_path)))
        os.remove(_get_relative_path(__file__, train_out_path))
        os.remove(_get_relative_path(__file__, test_out_path))

    def test_evaluate(self):
        model_out_path = 'word2vec_model.tsv'
        cr_out_path = 'word2vec_cr.txt'
        cm_out_path = 'word2vec.png'

        argv_embeddings = [
            __file__,
            'get-entity-embeddings',
            '--method', 'word2vec',
            '--train-data-path', _get_relative_path(__file__, '../mock_data/train.txt'),
            '--output-dir', _get_relative_path(__file__, '.'),
        ]
        main(argv_embeddings)

        argv_evaluate = [
            __file__,
            'evaluate',
            '--method', 'word2vec',
            '--train-data-path', _get_relative_path(__file__, '../mock_data/train.txt'),
            '--test-data-path', _get_relative_path(__file__, '../mock_data/test.txt'),
            '--embeddings-path', _get_relative_path(__file__, 'word2vec_model.tsv'),
            '--results-path', _get_relative_path(__file__, '.'),
        ]
        main(argv_evaluate)

        self.assertTrue(os.path.exists(_get_relative_path(__file__, model_out_path)))
        self.assertTrue(os.path.exists(_get_relative_path(__file__, cr_out_path)))
        self.assertTrue(os.path.exists(_get_relative_path(__file__, cm_out_path)))
        os.remove(_get_relative_path(__file__, model_out_path))
        os.remove(_get_relative_path(__file__, cr_out_path))
        os.remove(_get_relative_path(__file__, cm_out_path))
