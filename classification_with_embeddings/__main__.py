import argparse
import json
import os.path
import sys

from gensim.models import Doc2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from classification_with_embeddings import Tasks, EntityEmbeddingMethod, InternalClassifier
from classification_with_embeddings import logger
from classification_with_embeddings.embedding import embed
from classification_with_embeddings.embedding.embed_util import get_word_to_embedding
from classification_with_embeddings.evaluation.evaluate import evaluate
from classification_with_embeddings.evaluation.get_clf import get_clf_with_internal_clf, get_clf_starspace, get_clf_with_internal_clf_doc2vec, get_clf_with_internal_clf_gs
from classification_with_embeddings.evaluation.train_test_split import get_train_test_split
from classification_with_embeddings.util.argparse import file_path, dir_path, proportion_float


def main(argv=None):
    if argv is None:
        argv = sys.argv

    # parse arguments
    parser = argparse.ArgumentParser(prog='classification-with-embeddings')
    subparsers = parser.add_subparsers(required=True, dest='task', help='Task to run')

    # COMPUTING DOCUMENT EMBEDDINGS FROM DOCUMENTS IN fastText FORMAT
    get_entity_embeddings_parser = subparsers.add_parser(Tasks.GET_ENTITY_EMBEDDINGS.value)
    get_entity_embeddings_parser.add_argument('--method', type=str, default=EntityEmbeddingMethod.STARSPACE.value,
                                              choices=[v.value for v in EntityEmbeddingMethod], help='Method of generating entity embeddings')
    get_entity_embeddings_parser.add_argument('--train-data-path', type=file_path, required=True, help='Path to file containing the training data in fastText format')
    get_entity_embeddings_parser.add_argument('--output-dir', type=dir_path, default='.', help='Path to directory in which to save the embeddings')
    get_entity_embeddings_parser.add_argument('--starspace-path', type=file_path, help='Path to StarSpace executable')
    get_entity_embeddings_parser.add_argument('--starspace-args', type=str, default='',
                                              help='Arguments passed to StarSpace implementation (key-value pairs such as val=1 enclosed in quotes with no commas separated by spaces)')
    get_entity_embeddings_parser.add_argument('--word2vec-args', type=str, default='',
                                              help='Arguments passed to Word2Vec implementation (key-value pairs such as val=1 enclosed in quotes with no commas separated by spaces)')
    get_entity_embeddings_parser.add_argument('--fasttext-args', type=str, default='',
                                              help='Arguments passed to fastText implementation (key-value pairs such as val=1 enclosed in quotes with no commas separated by spaces)')
    get_entity_embeddings_parser.add_argument('--doc2vec-args', type=str, default='',
                                              help='Arguments passed to Doc2Vec implementation (key-value pairs such as val=1 enclosed in quotes with no commas separated by spaces)')

    # DATA TRAIN-TEST SPLIT
    train_test_split_parser = subparsers.add_parser(Tasks.TRAIN_TEST_SPLIT.value)
    train_test_split_parser.add_argument('--data-path', type=file_path, required=True, help='Path to file containing the data in fastText format')
    train_test_split_parser.add_argument('--train-size', type=proportion_float, default=0.8, help='Proportion of the dataset to include in the train split')
    train_test_split_parser.add_argument('--no-stratify', action='store_true', help='Do not split the data in a stratified fashion')
    train_test_split_parser.add_argument('--output-dir', type=dir_path, default='.', help='Path to directory in which to save the resulting files containing the training and test data')

    # DOCUMENT EMBEDDING EVALUATION USING CLASSIFICATION
    evaluate_parser = subparsers.add_parser(Tasks.EVALUATE.value)
    evaluate_parser.add_argument('--method', type=str, required=True,
                                 choices=[v.value for v in EntityEmbeddingMethod], help='Entity embedding method to evaluate')
    evaluate_parser.add_argument('--train-data-path', type=file_path, required=True, help='Path to file containing the training data in fastText format (for training internal classifiers)')
    evaluate_parser.add_argument('--test-data-path', type=file_path, required=True, help='Path to file containing the test data in fastText format')
    evaluate_parser.add_argument('--validation-size', type=proportion_float, default=0.3, help='Proportion of the dataset to use for hyperparameter tuning')
    evaluate_parser.add_argument('--no-stratify', action='store_true', help='Do not split the data in a stratified fashion')
    evaluate_parser.add_argument('--param-grid-path', type=file_path, help='Path to parameter grid in JSON format')
    evaluate_parser.add_argument('--cv', type=int, default=5, help='Number of folds to use when doing cross-validation')
    evaluate_parser.add_argument('--embeddings-path', type=file_path, help='Path to stored feature embeddings')
    evaluate_parser.add_argument('--doc2vec-model-path', type=file_path, help='Path to stored Doc2Vec model')
    evaluate_parser.add_argument('--binary', action='store_true', help='Embeddings are stored in binary format')
    evaluate_parser.add_argument('--results-path', type=dir_path, default='.', help='Path to directory in which to save the results')
    evaluate_parser.add_argument('--internal-clf', type=str, default=InternalClassifier.RANDOM_FOREST.value, help='Internal classifier to use (if applicable)')
    evaluate_parser.add_argument('--internal-clf-args', type=str, default='',
                                 help='Arguments passed to internal classifier if applicable (key-value pairs such as val=1 enclose in quotes with no commas separated by spaces)')

    parsed_args = vars(parser.parse_args(argv[1:]))

    if parsed_args['task'] == Tasks.GET_ENTITY_EMBEDDINGS.value:
        # COMPUTING DOCUMENT EMBEDDINGS FROM fastText FORMAT INPUT
        logger.info('Obtaining entity embeddings.')
        task_get_entity_embeddings(parsed_args)
    elif parsed_args['task'] == Tasks.TRAIN_TEST_SPLIT.value:
        # OBTAINING FILES CORRESPONDING TO A TRAIN-TEST SPLIT
        logger.info('Performing train-test split.')
        task_train_test_split(parsed_args)
    elif parsed_args['task'] == Tasks.EVALUATE.value:
        # EVALUATING EMBEDDING-BASED CLASSIFIERS
        logger.info('Performing evaluation.')
        task_evaluate(parsed_args)
    else:
        raise NotImplementedError('Task {0} not implemented.'.format(parsed_args['task']))


def task_get_entity_embeddings(parsed_args: dict):
    if parsed_args['method'] == EntityEmbeddingMethod.STARSPACE.value:
        # STARSPACE
        logger.info('Using StarSpace method.')
        if parsed_args['starspace_path'] is None:
            raise ValueError('path to StarSpace executable must be defined when using StarSpace')
        embed.get_starspace_embeddings(parsed_args['starspace_path'], parsed_args['train_data_path'], parsed_args['output_dir'], parsed_args['starspace_args'])
    elif parsed_args['method'] == EntityEmbeddingMethod.WORD2VEC.value:
        # WORD2VEC
        logger.info('Using Word2Vec method.')
        embed.get_word2vec_embeddings(parsed_args['train_data_path'], parsed_args['output_dir'], parsed_args['word2vec_args'])
    elif parsed_args['method'] == EntityEmbeddingMethod.FASTTEXT.value:
        # FASTTEXT
        logger.info('Using fastText method.')
        embed.get_fasttext_embeddings(parsed_args['train_data_path'], parsed_args['output_dir'], parsed_args['fasttext_args'])
    elif parsed_args['method'] == EntityEmbeddingMethod.DOC2VEC.value:
        # DOC2VEC
        logger.info('Using Doc2Vec method.')
        embed.get_doc2vec_embeddings(parsed_args['train_data_path'], parsed_args['output_dir'], parsed_args['doc2vec_args'])
    else:
        raise NotImplementedError('Method {0} not implemented'.format(parsed_args['method']))


def task_train_test_split(parsed_args: dict):
    get_train_test_split(parsed_args['data_path'], parsed_args['output_dir'], parsed_args['train_size'], not parsed_args['no_stratify'])


def task_evaluate(parsed_args: dict):
    logger.info('Performing evaluation of method \'{0}\''.format(parsed_args['method']))

    if parsed_args['embeddings_path'] is not None or (parsed_args['doc2vec_model_path'] is not None and parsed_args['method'] == 'doc2vec'):
        if parsed_args['method'] == EntityEmbeddingMethod.STARSPACE.value:
            # STARSPACE
            word_to_embedding = get_word_to_embedding(parsed_args['embeddings_path'], binary=parsed_args['binary'])
            clf = get_clf_starspace(word_to_embedding)
        elif parsed_args['method'] == EntityEmbeddingMethod.WORD2VEC.value or \
                parsed_args['method'] == EntityEmbeddingMethod.FASTTEXT.value or \
                parsed_args['method'] == EntityEmbeddingMethod.PRE_TRAINED_FROM_FILE.value:
            # STORED EMBEDDINGS
            word_to_embedding = get_word_to_embedding(parsed_args['embeddings_path'], binary=parsed_args['binary'])
            clf_internal = _get_clf_internal(parsed_args['internal_clf'])
            clf = get_clf_with_internal_clf(word_to_embedding, parsed_args['train_data_path'], clf_internal, parsed_args['internal_clf_args'])
        elif parsed_args['method'] == EntityEmbeddingMethod.DOC2VEC.value:
            clf_internal = _get_clf_internal(parsed_args['internal_clf'])
            doc2vec_model = Doc2Vec.load(parsed_args['doc2vec_model_path'])
            clf = get_clf_with_internal_clf_doc2vec(doc2vec_model, parsed_args['train_data_path'], clf_internal, parsed_args['internal_clf_args'])
        else:
            raise NotImplementedError('Method \'{0}\' not implemented.'.format(parsed_args['method']))
    else:
        if parsed_args['method'] == 'starspace':
            raise NotImplementedError('Method \'{}\' not supported when using grid search.')

        get_train_test_split(parsed_args['train_data_path'], '.', parsed_args['validation_size'], not parsed_args['no_stratify'], train_suffix='gs_train', test_suffix='gs_val')
        param_grid = _parse_param_grid(parsed_args['param_grid_path'])
        clf_internal = _get_clf_internal(parsed_args['internal_clf'])

        clf = get_clf_with_internal_clf_gs(
            train_data_path=os.path.join('.', _add_suffix_to_file_path(parsed_args['train_data_path'], '_gs_train')),
            validation_data_path=os.path.join('.', _add_suffix_to_file_path(parsed_args['train_data_path'], '_gs_val')),
            param_grid=param_grid,
            embedding_method=parsed_args['method'],
            clf_internal=clf_internal,
            cv=parsed_args['cv']
        )

    # evaluate classifier
    evaluate(clf, parsed_args['method'], parsed_args['test_data_path'], parsed_args['results_path'])


def _add_suffix_to_file_path(file_path: str, suffix: str) -> str:
    name, ext = os.path.splitext(os.path.basename(file_path))
    return name + suffix + ext


def _parse_param_grid(param_grid_path: str) -> dict:
    with open(param_grid_path, 'r') as f:
        return json.loads(f.read())


def _get_clf_internal(internal_clf_kind: str):
    # set internal classifier
    if internal_clf_kind == InternalClassifier.LOGISTIC_REGRESSION.value:
        return LogisticRegression
    elif internal_clf_kind == InternalClassifier.RANDOM_FOREST.value:
        return RandomForestClassifier
    elif internal_clf_kind == InternalClassifier.SVC.value:
        return SVC
    else:
        raise NotImplementedError('Classifier {0} not implemented.'.format(internal_clf_kind))


if __name__ == '__main__':
    sys.exit(main(sys.argv))
