from typing import Dict

import gensim
import numpy as np

from classification_with_embeddings import LABEL_WORD_PREFIX
from classification_with_embeddings.embedding import logger


def get_aggregate_embedding(features: str, word_to_embedding: Dict[str, np.ndarray[1, ...]], method='average') -> np.ndarray[1, ...]:
    """get embedding for a new set of features (new document).

    :param features: features in fastText format
    :param word_to_embedding: mapping of words to their embeddings
    :param method: word embedding aggregation method to use
    :return: aggregate vector composed of individual entity embeddings
    """

    words = [w for w in features.split() if LABEL_WORD_PREFIX not in w]
    if method == 'average':
        return _get_aggregate_embedding_average(words, word_to_embedding)
    else:
        raise ValueError('method {} not supported'.format(method))


def _get_aggregate_embedding_average(words: list[str], word_to_embedding: dict) -> np.ndarray[1, ...]:
    """Get aggregate embedding by averaging constituent word embeddings.

    :param words: list of constituent words
    :param word_to_embedding: mapping of words to their embeddings
    :return: aggregate vector composed of individual entity embeddings
    """
    emb_len = len(next(iter(word_to_embedding.values())))
    aggregate_emb = np.zeros(emb_len, dtype=float)
    count = 0
    for word in words:
        if word in word_to_embedding:
            aggregate_emb += word_to_embedding[word]
            count += 1
    if count > 0:
        aggregate_emb /= count
    return aggregate_emb


def get_word_to_embedding(path_to_embeddings: str, binary: bool = False) -> Dict[str, np.ndarray[1, ...]]:
    """get dictionary mapping words to their embeddings.

    :param path_to_embeddings: path embeddings
    :param binary: are the embeddings stored in binary format or not
    :return: dictionary mapping words to their embeddings
    """

    logger.info('Obtaining mapping from words to their embeddings.')

    if binary:
        res = gensim.models.KeyedVectors.load_word2vec_format(path_to_embeddings, binary=True)
        return {k: res[k] for k in res.index_to_key}
    else:
        with open(path_to_embeddings, 'r') as f:
            word_to_embedding = dict()
            for emb in f:
                emb_l = emb.strip().split('\t')
                word_to_embedding[emb_l[0]] = np.asarray(emb_l[1:], dtype=float)
            return word_to_embedding
