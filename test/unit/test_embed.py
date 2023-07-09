import glob
import os
import unittest
from typing import Final

from classification_with_embeddings.embedding.embed import (
    get_word2vec_embeddings,
    get_fasttext_embeddings,
    get_starspace_embeddings,
    get_doc2vec_embeddings,
    get_doc_embedder_instance
)
from classification_with_embeddings.embedding.embed_util import (
    get_aggregate_embedding,
    get_word_to_embedding,
    get_aggregate_embeddings
)
from test.test_utils import get_relative_path


class TestEmbed(unittest.TestCase):
    N_WORD_EMBEDDINGS: Final = 13

    def test_get_starspace_embeddings(self):
        get_starspace_embeddings(get_relative_path(__file__, '../../embedding_methods/StarSpace/starspace'), '../mock_data/data.txt', get_relative_path(__file__, '.'), '')
        self._assert_and_delete_created_embeddings_file(get_relative_path(__file__, 'starspace_model.tsv'), self.N_WORD_EMBEDDINGS + 2)
        os.remove(get_relative_path(__file__, 'starspace_model'))

    def test_get_word2vec_embeddings(self):
        get_word2vec_embeddings(get_relative_path(__file__, '../mock_data/data.txt'), get_relative_path(__file__, '.'), '')
        self._assert_and_delete_created_embeddings_file(get_relative_path(__file__, 'word2vec_model.tsv'), self.N_WORD_EMBEDDINGS)

    def test_get_fasttext_embeddings(self):
        get_fasttext_embeddings(get_relative_path(__file__, '../mock_data/data.txt'), get_relative_path(__file__, '.'), '')
        self._assert_and_delete_created_embeddings_file(get_relative_path(__file__, 'fasttext_model.tsv'), self.N_WORD_EMBEDDINGS)

    def test_get_doc2vec_embeddings(self):
        get_doc2vec_embeddings(get_relative_path(__file__, '../mock_data/data.txt'), get_relative_path(__file__, '.'), '')

        file_path = get_relative_path(__file__, 'doc2vec_model.bin')
        self.assertTrue(os.path.exists(file_path))
        os.remove(file_path)

    def test_get_word_to_embedding(self):
        word_to_embedding = get_word_to_embedding(get_relative_path(__file__, '../mock_data/mock_model.tsv'))
        self.assertEqual(['this', 'is', 'a', 'test'], list(word_to_embedding.keys()))
        self.assertEqual([0.1, 0.2, 0.3], list(word_to_embedding['this']))
        self.assertEqual([0.9, 0.3, 0.5], list(word_to_embedding['is']))
        self.assertEqual([0.0, 0.1, 0.7], list(word_to_embedding['a']))
        self.assertEqual([1.0, 0.9, 0.1], list(word_to_embedding['test']))

    def test_get_aggregate_embedding(self):
        word_to_embedding = get_word_to_embedding(get_relative_path(__file__, '../mock_data/mock_model.tsv'))

        aggregate_emb1 = get_aggregate_embedding(['something'], word_to_embedding)
        self.assertEqual([0, 0, 0], list(aggregate_emb1))

        aggregate_emb2 = get_aggregate_embedding(['this', 'something'], word_to_embedding)
        self.assertEqual([0.1, 0.2, 0.3], list(aggregate_emb2))

        aggregate_emb3 = get_aggregate_embedding(['this', 'something', 'is', 'nothing'], word_to_embedding)
        self.assertEqual([1.0 / 2.0, 0.5 / 2.0, 0.8 / 2.0], list(aggregate_emb3))

    def test_get_aggregate_embeddings(self):
        word_to_embedding = get_word_to_embedding(get_relative_path(__file__, '../mock_data/mock_model.tsv'))

        aggregate_embs = get_aggregate_embeddings([['something'], ['this', 'something'], ['this', 'something', 'is', 'nothing']], word_to_embedding)
        self.assertEqual([[0, 0, 0], [0.1, 0.2, 0.3], [1.0 / 2.0, 0.5 / 2.0, 0.8 / 2.0]], aggregate_embs.tolist())

    def test_get_doc_embedder_instance(self):
        doc_embedder = get_doc_embedder_instance('word2vec', get_relative_path(__file__, '../mock_data/sanity_check_dataset_train.txt'))
        self.assertIsNotNone(doc_embedder)

        res = doc_embedder.transform([['pear', 'apple'], ['cat', 'hedgehog']])
        self.assertEqual((2, 100), res.shape)

        res = doc_embedder([['pear', 'apple'], ['cat', 'hedgehog']])
        self.assertEqual((2, 100), res.shape)

    def _assert_and_delete_created_embeddings_file(self, file_path: str, n_embeddings: int):
        # test saved file contents correct
        files = glob.glob(file_path)
        self.assertEqual(1, len(files))
        saved_file_path = files[0]
        with open(saved_file_path, 'r') as f:
            self.assertEqual(n_embeddings, len(f.readlines()))

        os.remove(saved_file_path)
