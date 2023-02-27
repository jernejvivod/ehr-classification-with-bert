import os
import unittest

from classification_with_embeddings.evaluation.train_test_split import get_train_test_split
from test.test_utils import get_relative_path


class TestTrainTestSplit(unittest.TestCase):
    def test_get_train_test_split(self):
        mock_data_path = '../mock_data/data_10_rows.txt'
        train_out_path = 'data_10_rows_train.txt'
        test_out_path = 'data_10_rows_test.txt'

        get_train_test_split(get_relative_path(__file__, mock_data_path), get_relative_path(__file__, "."))
        with open(get_relative_path(__file__, train_out_path)) as f_train:
            lines_train = f_train.readlines()
            self.assertEqual(8, len(lines_train))
        with open(get_relative_path(__file__, test_out_path)) as f_test:
            lines_test = f_test.readlines()
            self.assertEqual(2, len(lines_test))

        self.assertEqual(0, len(set(lines_test).intersection(set(lines_train))))

        os.remove(get_relative_path(__file__, train_out_path))
        os.remove(get_relative_path(__file__, test_out_path))
