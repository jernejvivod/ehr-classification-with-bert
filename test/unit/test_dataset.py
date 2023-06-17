import unittest

from classification_with_embeddings.embedding.cnn.dataset import FastTextFormatDataset
from test.test_utils import get_relative_path


class TestEmbed(unittest.TestCase):
    def test_fasttext_format_dataset(self):
        dataset = FastTextFormatDataset(get_relative_path(__file__, '../mock_data/data.txt'))

        ex_0 = dataset[0]
        ex_1 = dataset[1]
        ex_2 = dataset[2]

        self.assertEqual(['this', 'is', 'a', 'simple', 'test'], ex_0[0])
        self.assertEqual(0, ex_0[1])

        self.assertEqual(['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog'], ex_1[0])
        self.assertEqual(1, ex_1[1])

        self.assertIsNone(ex_2)
