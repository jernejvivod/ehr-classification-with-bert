import os
import subprocess

from text_classification_with_embeddings.util.errors import EmbeddingError


def get_starspace_entity_embeddings(starspace_path: str, train_data_path: str, output_dir: str, starspace_args: str):
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

    print('Entity embeddings saved to {0}'.format(model_path))


def get_word2vec_embeddings(train_data_path: str, output_dir: str):
    """Get entity embeddings using Word2Vec.

    :param train_data_path: path to training data in fastText format
    :param output_dir: path to directory in which to store the embeddings
    """
    pass
