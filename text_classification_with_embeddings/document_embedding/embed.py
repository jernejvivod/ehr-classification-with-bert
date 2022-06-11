import os
import subprocess

import numpy as np
from gensim.models import Word2Vec, FastText

from text_classification_with_embeddings import LABEL_WORD_PREFIX
from text_classification_with_embeddings.document_embedding import logger
from text_classification_with_embeddings.util.arguments import process_param_spec
from text_classification_with_embeddings.util.errors import EmbeddingError
from text_classification_with_embeddings.util.iterators import SentenceIteratorFastTextFormat


def get_starspace_entity_embeddings(starspace_path: str, train_data_path: str, output_dir: str, starspace_args: str) -> None:
    """Get entity embeddings using StarSpace.

    :param starspace_path: path to StarSpace executable
    :param train_data_path: path to training data in fastText format
    :param output_dir: path to directory in which to store the embeddings
    :param starspace_args: arguments passed to StarSpace implementation
    """

    # get StarSpace entity embeddings
    model_path = os.path.join(os.path.abspath(output_dir), 'starspace_model.tsv')
    p = subprocess.run(
        [starspace_path, 'train', '-trainFile', train_data_path, '-model', os.path.join(os.path.abspath(output_dir), 'starspace_model')] + starspace_args.split()
    )
    if p.returncode != 0:
        raise EmbeddingError('StarSpace', p.returncode)

    logger.info('Successfuly saved StarSpace embeddings')
    print('Entity embeddings saved to {0}'.format(model_path))


def get_word2vec_embeddings(train_data_path: str, output_dir: str, word2vec_args: str) -> None:
    """Get entity embeddings using Word2Vec.

    :param train_data_path: path to training data in fastText format
    :param output_dir: path to directory in which to store the embeddings
    :param word2vec_args: arguments passed to Word2Vec implementation
    """

    # compute and save entity embeddings
    with SentenceIteratorFastTextFormat(train_data_path) as sent_it:

        # if custom arguments specified, process
        if len(word2vec_args) > 0:
            params = process_param_spec(word2vec_args)
        else:
            params = {'vector_size': 100, 'window': 5, 'min_count': 1, 'workers': 4}

        # train model
        w2v_model = Word2Vec(sentences=sent_it, **params)

        # save embeddings in TSV format
        logger.info('Saving computed Word2Vec embeddings')
        output_file_name = 'word2vec_model.tsv'
        with open(os.path.join(output_dir, output_file_name), 'w') as f:
            for idx, emb in enumerate(w2v_model.wv.vectors):
                key = w2v_model.wv.index_to_key[idx]
                f.write(key + '\t' + '\t'.join(map(str, emb)) + '\n')

    print('Entity embeddings saved to {0}'.format(output_dir + output_file_name))


def get_fasttext_embeddings(train_data_path: str, output_dir: str, fasttext_args: str) -> None:
    """Get entity embeddings using fastText.

    :param train_data_path: path to training data in fastText format
    :param output_dir: path to directory in which to store the embeddings
    :param fasttext_args: arguments passed to fastText implementation
    """
    # compute and save entity embeddings
    with SentenceIteratorFastTextFormat(train_data_path) as sent_it:

        # if custom arguments specified, process
        if len(fasttext_args) > 0:
            params = process_param_spec(fasttext_args)
        else:
            params = {'vector_size': 10, 'window': 3, 'min_count': 1}

        # train model
        ft_model = FastText(**params)
        ft_model.build_vocab(corpus_iterable=sent_it)
        ft_model.train(corpus_iterable=sent_it, total_examples=len(sent_it), epochs=10)

        # save embeddings in TSV format
        logger.info('Saving computed fastText embeddings')
        output_file_name = 'fasttext_model.tsv'
        with open(os.path.join(output_dir, output_file_name), 'w') as f:
            for idx, emb in enumerate(ft_model.wv.vectors):
                key = ft_model.wv.index_to_key[idx]
                f.write(key + '\t' + '\t'.join(map(str, emb)) + '\n')

    print('Entity embeddings saved to {0}'.format(output_dir + output_file_name))


def get_aggregate_embedding(features: str, word_to_embedding) -> np.ndarray:
    """get embedding for a new set of features (new document).

    :param features: features in fastText format
    :param word_to_embedding: mapping of words to their embeddings
    :return: aggregate vector composed from individual entity embeddings
    """

    # construct aggregate embedding
    emb_len = len(next(iter(word_to_embedding.values())))
    words = [w for w in features.split() if LABEL_WORD_PREFIX not in w]
    aggregate_emb = np.zeros(emb_len, dtype=float)
    count = 0
    for word in words:
        if word in word_to_embedding:
            aggregate_emb += word_to_embedding[word]
            count += 1
    if count > 0:
        aggregate_emb /= count
    return aggregate_emb


def get_word_to_embedding(path_to_embeddings: str) -> dict:
    """get dictionary mapping words to their embeddings.

    :param path_to_embeddings: path embeddings
    :return: dictionary mapping words to their embeddings
    """

    logger.info('Obtaining mapping from words to their embeddings')
    with open(path_to_embeddings, 'r') as f:
        word_to_embedding = dict()
        for emb in f:
            emb_l = emb.strip().split('\t')
            word_to_embedding[emb_l[0]] = np.asarray(emb_l[1:], dtype=float)
        return word_to_embedding
