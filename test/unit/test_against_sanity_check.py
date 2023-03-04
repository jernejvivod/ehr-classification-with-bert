import os.path
import unittest
from typing import Final

from gensim.models import Doc2Vec

from classification_with_embeddings.embedding.embed import get_word2vec_embeddings, get_fasttext_embeddings, get_starspace_embeddings, get_doc2vec_embeddings
from classification_with_embeddings.embedding.embed_util import get_word_to_embedding
from classification_with_embeddings.evaluation.evaluate import evaluate
from classification_with_embeddings.evaluation.get_clf import get_clf_starspace, get_clf_with_internal_clf, get_clf_with_internal_clf_doc2vec
from test.test_utils import get_relative_path


class TestEmbed(unittest.TestCase):
    TRAINING_SET_PATH: Final = get_relative_path(__file__, '../mock_data/sanity_check_dataset_train.txt')
    TEST_SET_PATH: Final = get_relative_path(__file__, '../mock_data/sanity_check_dataset_test.txt')
    OUT_PATH: Final = get_relative_path(__file__, 'sanity_check_dataset_results/')
    STARSPACE_PATH: Final = get_relative_path(__file__, '../../embedding_methods/StarSpace/starspace')
    BIO_WORD_VEC_PATH: Final = get_relative_path(__file__, '../../embedding_methods/BioWordVec/bio_embedding_extrinsic')

    def test_classification_sanity_check(self):
        self._run_test_classification_sanity_check(method='word2vec')
        self._run_test_classification_sanity_check(method='fasttext')
        self._run_test_classification_sanity_check(method='doc2vec')
        self._run_test_classification_sanity_check(method='starspace')
        self._run_test_classification_sanity_check(method='pre-trained-from-file')

    def _run_test_classification_sanity_check(self, method: str, args: str = '', clf_internal=None, internal_clf_args: str = ''):
        if method == 'word2vec':
            get_word2vec_embeddings(self.TRAINING_SET_PATH, self.OUT_PATH, args)
            word_to_embedding = get_word_to_embedding(get_relative_path(__file__, os.path.join(self.OUT_PATH, '{}_model.tsv'.format(method))))
            clf = get_clf_with_internal_clf(word_to_embedding, self.TRAINING_SET_PATH, clf_internal=clf_internal, internal_clf_args=internal_clf_args)
        elif method == 'fasttext':
            get_fasttext_embeddings(self.TRAINING_SET_PATH, self.OUT_PATH, args)
            word_to_embedding = get_word_to_embedding(get_relative_path(__file__, os.path.join(self.OUT_PATH, '{}_model.tsv'.format(method))))
            clf = get_clf_with_internal_clf(word_to_embedding, self.TRAINING_SET_PATH, clf_internal=clf_internal, internal_clf_args=internal_clf_args)
        elif method == 'doc2vec':
            get_doc2vec_embeddings(self.TRAINING_SET_PATH, self.OUT_PATH, args)
            doc2vec_model = Doc2Vec.load(get_relative_path(__file__, os.path.join(self.OUT_PATH, '../mock_data/doc2vec_model.bin')))
            clf = get_clf_with_internal_clf_doc2vec(doc2vec_model, self.TRAINING_SET_PATH, clf_internal=clf_internal, internal_clf_args=internal_clf_args)
        elif method == 'starspace':
            get_starspace_embeddings(self.STARSPACE_PATH, self.TRAINING_SET_PATH, self.OUT_PATH, args)
            word_to_embedding = get_word_to_embedding(get_relative_path(__file__, os.path.join(self.OUT_PATH, '{}_model.tsv'.format(method))))
            clf = get_clf_starspace(word_to_embedding)
        elif method == 'pre-trained-from-file':
            word_to_embedding = get_word_to_embedding(self.BIO_WORD_VEC_PATH, binary=True)
            clf = get_clf_with_internal_clf(word_to_embedding, self.TRAINING_SET_PATH, clf_internal=clf_internal, internal_clf_args=internal_clf_args)
        else:
            raise NotImplementedError('Unknown method {}.'.format(method))

        evaluate(clf, method, get_relative_path(__file__, self.TEST_SET_PATH), self.OUT_PATH)
