import os
import subprocess

from gensim.models import Word2Vec

from text_classification_with_embeddings.util.errors import EmbeddingError
from text_classification_with_embeddings.util.generators import SentenceIteratorFastTextFormat


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


def get_word2vec_embeddings(train_data_path: str, output_dir: str, word2vec_args: str):
    """Get entity embeddings using Word2Vec.

    :param train_data_path: path to training data in fastText format
    :param output_dir: path to directory in which to store the embeddings
    :param word2vec_args: arguments passed to Word2Vec implementation
    """

    # compute and save entity embeddings
    with SentenceIteratorFastTextFormat(train_data_path) as sent_it:

        # if custom arguments specified, process
        if len(word2vec_args) > 0:
            custom_params_split = {kv[0]: kv[1] for exp in word2vec_args.split() for kv in [exp.split('=')]}
            params = dict()
            for k, v in custom_params_split.items():
                try:
                    params[k] = int(v)
                except ValueError:
                    try:
                        params[k] = float(v)
                    except ValueError:
                        params[k] = v

            # train with custom parameters
            w2v_model = Word2Vec(sentences=sent_it, **params)
        else:
            w2v_model = Word2Vec(sentences=sent_it, vector_size=100, window=5, min_count=1, workers=4)

        # save embeddings in TSV format
        with open(os.path.join(output_dir, 'word2vec_model.tsv'), 'w') as f:
            for idx, emb in enumerate(w2v_model.wv.vectors):
                key = w2v_model.wv.index_to_key[idx]
                f.write(key + '\t' + '\t'.join(map(str, emb)) + '\n')
