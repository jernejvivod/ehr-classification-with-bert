import unittest

from classification_with_embeddings.embedding.cnn.train import train_cnn_model
from test.test_utils import get_relative_path


class TestTrainCnnModel(unittest.TestCase):
    def test_train_cnn_model(self):
        train_cnn_model(
            train_data_path=get_relative_path(__file__, '../mock_data/sanity_check_dataset_train.txt'),
            word_embeddings_path=get_relative_path(__file__, '../mock_data/sanity_check_dataset_train_word2vec_model.tsv'),
            n_labels=2
        )
