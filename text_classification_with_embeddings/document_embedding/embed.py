def get_starspace_entity_embeddings(starspace_path: str, train_data_path: str, output_dir: str, **starspace_args):
    """Get entity embeddings using StarSpace.

    :param starspace_path: path to StarSpace executable
    :param train_data_path: path to training data in fastText format
    :param output_dir: path to directory in which to store the embeddings
    :param starspace_args: arguments passed to StarSpace implementation
    """
    print()
    pass


def get_word2vec_embeddings(train_data_path: str, output_dir: str):
    """Get entity embeddings using Word2Vec.

    :param train_data_path: path to training data in fastText format
    :param output_dir: path to directory in which to store the embeddings
    """
    pass
