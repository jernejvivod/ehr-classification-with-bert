import os
import subprocess

from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from classification_with_embeddings.evaluation.embedders.a_doc_embedder import ADocEmbedder
from classification_with_embeddings.util.arguments import process_param_spec
from classification_with_embeddings.util.errors import EmbeddingError
from classification_with_embeddings.util.iterators import SentenceIteratorFastTextFormat


def get_starspace_embeddings(starspace_path: str, train_data_path: str, output_dir: str, starspace_args: str) -> str:
    """Get entity embeddings using StarSpace.

    :param starspace_path: path to StarSpace executable
    :param train_data_path: path to training data in fastText format
    :param output_dir: path to directory in which to store the embeddings
    :param starspace_args: arguments passed to StarSpace implementation
    """

    model_path = os.path.join(os.path.abspath(output_dir), 'starspace_model')
    p = subprocess.run(
        [starspace_path, 'train', '-trainFile', train_data_path, '-model', model_path] + starspace_args.split()
    )
    if p.returncode != 0:
        raise EmbeddingError('StarSpace', p.returncode)

    return model_path


def get_word2vec_embeddings(train_data_path: str, output_dir: str, word2vec_args: str) -> str:
    """Get entity embeddings using Word2Vec.

    :param train_data_path: path to training data in fastText format
    :param output_dir: path to directory in which to store the embeddings
    :param word2vec_args: arguments passed to Word2Vec implementation
    """

    with SentenceIteratorFastTextFormat(train_data_path) as sent_it:
        model_params = _parse_params_or_get_default(
            word2vec_args,
            default_params={'vector_size': 100, 'window': 5, 'min_count': 1, 'workers': 4}
        )
        w2v_model = Word2Vec(sentences=sent_it, **model_params)

        # save embeddings in TSV format
        output_file_name = 'word2vec_model.tsv'
        out_path = _save_wv_to_file(w2v_model, output_dir, output_file_name)

    return out_path


def get_fasttext_embeddings(train_data_path: str, output_dir: str, fasttext_args: str) -> str:
    """Get entity embeddings using fastText.

    :param train_data_path: path to training data in fastText format
    :param output_dir: path to directory in which to store the embeddings
    :param fasttext_args: arguments passed to fastText implementation
    """
    with SentenceIteratorFastTextFormat(train_data_path) as sent_it:
        model_params = _parse_params_or_get_default(
            fasttext_args,
            {'vector_size': 10, 'window': 3, 'min_count': 1}
        )
        ft_model = FastText(**model_params)
        ft_model.build_vocab(corpus_iterable=sent_it)
        ft_model.train(corpus_iterable=sent_it, total_examples=len(sent_it), epochs=10)

        # save embeddings in TSV format
        output_file_name = 'fasttext_model.tsv'
        out_path = _save_wv_to_file(ft_model, output_dir, output_file_name)

    return out_path


def get_doc2vec_embeddings(train_data_path: str, output_dir: str, doc2vec_args: str) -> str:
    """Get entity embeddings using Doc2Vec.

    :param train_data_path: path to training data in fastText format
    :param output_dir: path to directory in which to store the embeddings
    :param doc2vec_args: arguments passed to fastText implementation
    """
    with SentenceIteratorFastTextFormat(train_data_path) as sent_it:
        tagged_data = [TaggedDocument(words=s, tags=[str(i)]) for i, s in enumerate(sent_it)]
        model_params = _parse_params_or_get_default(
            doc2vec_args,
            {'vector_size': 100, 'window': 2, 'min_count': 1, 'epochs': 10}
        )
        model = Doc2Vec(tagged_data, **model_params)

        # save model
        output_file_name = 'doc2vec_model.bin'
        out_path = os.path.join(output_dir, output_file_name)
        model.save(out_path)

    return out_path


def get_doc_embedder_instance(method: str, train_data_path: str, method_args: str = "", **a_doc_embedder_kwargs) -> ADocEmbedder:
    """get ADocEmbedder instance trained on specified training file

    :param method: embedding method to use ('word2vec', 'fasttext', 'doc2vec', or 'starspace')
    :param train_data_path: path to training data in fastText format
    :param method_args: arguments passed to embedding implementation
    :param a_doc_embedder_kwargs: additional keyword arguments to pass to the ADocEmbedder instance constructor
    """

    with SentenceIteratorFastTextFormat(train_data_path) as sent_it:
        model_params = _parse_params_or_get_default(method_args, {})

        # get model
        doc_embedder = ADocEmbedder.factory(method, **a_doc_embedder_kwargs)
        doc_embedder.method_kwargs = model_params
        doc_embedder.fit(sent_it, None)

    return doc_embedder


def _parse_params_or_get_default(model_params: str, default_params: dict) -> dict:
    """Parse embedding model arguments specified as string containing key-value pairs or return provided
    defaults if none specified.

    :param model_params: provided model params in the form of a str containing key-value pairs
    :param default_params: dict specifying the default parameters if model_params is an empty string
    :return: dict mapping the parsed or default parameter names to their values
    """
    if len(model_params) > 0:
        return process_param_spec(model_params)
    else:
        return default_params


def _save_wv_to_file(model, output_dir: str, output_file_name: str) -> str:
    """Save word vectors computed using a gensim model to a file.

    :param model: trained gensim model
    :param output_dir: output directory
    :param output_file_name: output file name
    :return: path to output file
    """
    out_path = os.path.join(output_dir, output_file_name)
    with open(out_path, 'w') as f:
        for idx, emb in enumerate(model.wv.vectors):
            key = model.wv.index_to_key[idx]
            f.write(key + '\t' + '\t'.join(map(str, emb)) + '\n')
    return out_path
