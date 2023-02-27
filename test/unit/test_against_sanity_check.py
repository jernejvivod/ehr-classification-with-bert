import os.path
import unittest
from typing import Final

from classification_with_embeddings.embedding.embed import get_word_to_embedding, get_word2vec_embeddings, get_fasttext_embeddings, get_starspace_embeddings
from classification_with_embeddings.evaluation.evaluate import evaluate
from classification_with_embeddings.evaluation.get_clf import get_clf_starspace, get_clf_with_internal_clf
from test.test_utils import get_relative_path


class TestEmbed(unittest.TestCase):
    TRAINING_SET_PATH: Final = get_relative_path(__file__, '../mock_data/sanity_check_dataset_train.txt')
    TEST_SET_PATH: Final = get_relative_path(__file__, '../mock_data/sanity_check_dataset_test.txt')
    OUT_PATH: Final = get_relative_path(__file__, 'sanity_check_dataset_results/')
    STARSPACE_PATH: Final = get_relative_path(__file__, '../../embedding_methods/StarSpace/starspace')

    def test_classification_sanity_check(self):
        self._run_test_classification_sanity_check(method='word2vec')
        self._run_test_classification_sanity_check(method='fasttext')
        self._run_test_classification_sanity_check(method='starspace')

    def _run_test_classification_sanity_check(self, method: str, word2vec_args: str = '', clf_internal=None, internal_clf_args='', starspace_args=''):
        if method == 'word2vec':
            get_word2vec_embeddings(self.TRAINING_SET_PATH, self.OUT_PATH, word2vec_args)
            word_to_embedding = get_word_to_embedding(get_relative_path(__file__, os.path.join(self.OUT_PATH, '{}_model.tsv'.format(method))))
            clf = get_clf_with_internal_clf(word_to_embedding, self.TRAINING_SET_PATH, clf_internal=clf_internal, internal_clf_args=internal_clf_args)
        elif method == 'fasttext':
            get_fasttext_embeddings(self.TRAINING_SET_PATH, self.OUT_PATH, word2vec_args)
            word_to_embedding = get_word_to_embedding(get_relative_path(__file__, os.path.join(self.OUT_PATH, '{}_model.tsv'.format(method))))
            clf = get_clf_with_internal_clf(word_to_embedding, self.TRAINING_SET_PATH, clf_internal=clf_internal, internal_clf_args=internal_clf_args)
        elif method == 'starspace':
            get_starspace_embeddings(self.STARSPACE_PATH, self.TRAINING_SET_PATH, self.OUT_PATH, starspace_args)
            word_to_embedding = get_word_to_embedding(get_relative_path(__file__, os.path.join(self.OUT_PATH, '{}_model.tsv'.format(method))))
            clf = get_clf_starspace(word_to_embedding)
        else:
            raise NotImplementedError('Unknown method {}.'.format(method))

        evaluate(clf, method, get_relative_path(__file__, self.TEST_SET_PATH), self.OUT_PATH)
