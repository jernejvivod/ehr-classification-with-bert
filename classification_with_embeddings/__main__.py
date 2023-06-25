import argparse
import sys

import torch
from gensim.models import Doc2Vec
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from torch.utils.data import DataLoader

from classification_with_embeddings import Tasks, EntityEmbeddingMethod, InternalClassifier, torch_device
from classification_with_embeddings import logger
from classification_with_embeddings.embedding.cnn.dataset import FastTextFormatDataset
from classification_with_embeddings.embedding.cnn.train import train_cnn_model
from classification_with_embeddings.embedding.embed import get_starspace_embeddings, get_doc2vec_embeddings, \
    get_fasttext_embeddings, get_word2vec_embeddings
from classification_with_embeddings.embedding.embed_util import get_word_to_embedding
from classification_with_embeddings.evaluation.clf.a_classifier import AClassifier
from classification_with_embeddings.evaluation.evaluate import evaluate_embeddings_model, evaluate_cnn_model
from classification_with_embeddings.evaluation.get_clf import get_clf_with_internal_clf, get_clf_starspace, \
    get_clf_with_internal_clf_doc2vec, get_clf_with_internal_clf_gs
from classification_with_embeddings.train_test_split.train_test_split import get_train_test_split
from classification_with_embeddings.util.argparse import file_path, dir_path, proportion_float, positive_int
from classification_with_embeddings.util.arguments import get_train_and_val_paths_for_multiple_train_files, \
    parse_param_grid


def main(argv=None):
    if argv is None:
        argv = sys.argv

    parser = argparse.ArgumentParser(prog='classification-with-embeddings')
    subparsers = parser.add_subparsers(required=True, dest='task', help='Task to run')

    # add subparsers for tasks
    _add_subparser_for_get_entity_embeddings(subparsers)
    _add_subparser_for_train_cnn_model(subparsers)
    _add_subparser_for_get_train_test_split(subparsers)
    _add_subparser_for_evaluate_embeddings_model(subparsers)
    _add_subparser_for_evaluate_cnn_model(subparsers)

    # run task
    _run_task(vars(parser.parse_args(argv[1:])))


def _run_task(parsed_args: dict):
    if parsed_args['task'] == Tasks.GET_ENTITY_EMBEDDINGS.value:
        # COMPUTING DOCUMENT EMBEDDINGS FROM fastText FORMAT INPUT
        logger.info('Obtaining entity embeddings.')
        _task_get_entity_embeddings(parsed_args)
    elif parsed_args['task'] == Tasks.TRAIN_CNN_MODEL.value:
        # COMPUTING DOCUMENT EMBEDDINGS FROM fastText FORMAT INPUT
        logger.info('Training CNN-based model.')
        _task_train_cnn_model(parsed_args)
    elif parsed_args['task'] == Tasks.TRAIN_TEST_SPLIT.value:
        # OBTAINING FILES CORRESPONDING TO A TRAIN-TEST SPLIT
        logger.info('Performing train-test split.')
        _task_train_test_split(parsed_args)
    elif parsed_args['task'] == Tasks.EVALUATE_EMBEDDINGS_MODEL.value:
        # EVALUATING EMBEDDING-BASED CLASSIFIERS
        logger.info('Performing evaluation of an embeddings-based model.')
        _task_evaluate_embeddings_model(parsed_args)
    elif parsed_args['task'] == Tasks.EVALUATE_CNN_MODEL.value:
        # EVALUATING EMBEDDING-BASED CLASSIFIERS
        logger.info('Performing evaluation of a CNN-based model.')
        _task_evaluate_cnn_model(parsed_args)
    else:
        raise NotImplementedError('Task {0} not implemented.'.format(parsed_args['task']))


def _add_subparser_for_get_entity_embeddings(subparsers):
    get_entity_embeddings_parser = subparsers.add_parser(Tasks.GET_ENTITY_EMBEDDINGS.value)
    get_entity_embeddings_parser.add_argument('--method', type=str, default=EntityEmbeddingMethod.STARSPACE.value,
                                              choices=[v.value for v in EntityEmbeddingMethod],
                                              help='Method of generating entity embeddings')
    get_entity_embeddings_parser.add_argument('--train-data-path', type=file_path, required=True,
                                              help='Path to file containing the training data in FastText format')
    get_entity_embeddings_parser.add_argument('--output-dir', type=dir_path, default='.',
                                              help='Path to directory in which to save the embeddings')
    get_entity_embeddings_parser.add_argument('--starspace-path', type=file_path, help='Path to StarSpace executable')
    get_entity_embeddings_parser.add_argument('--starspace-args', type=str, default='',
                                              help='Arguments passed to StarSpace implementation (key-value pairs such'
                                                   ' as val=1 enclosed in quotes with no commas separated by spaces)')
    get_entity_embeddings_parser.add_argument('--word2vec-args', type=str, default='',
                                              help='Arguments passed to Word2Vec implementation (key-value pairs such'
                                                   ' as val=1 enclosed in quotes with no commas separated by spaces)')
    get_entity_embeddings_parser.add_argument('--fasttext-args', type=str, default='',
                                              help='Arguments passed to fastText implementation (key-value pairs such'
                                                   ' as val=1 enclosed in quotes with no commas separated by spaces)')
    get_entity_embeddings_parser.add_argument('--doc2vec-args', type=str, default='',
                                              help='Arguments passed to Doc2Vec implementation (key-value pairs such'
                                                   ' as val=1 enclosed in quotes with no commas separated by spaces)')


def _add_subparser_for_train_cnn_model(subparsers):
    train_cnn_model_parser = subparsers.add_parser(Tasks.TRAIN_CNN_MODEL.value)
    train_cnn_model_parser.add_argument('--train-data-path', type=file_path, required=True, nargs='+',
                                        action=UnnestSingletonListElement,
                                        help='Path to file containing the training data in FastText format')
    train_cnn_model_parser.add_argument('--word-embeddings-path', type=file_path, required=True,
                                        help='Path to file containing the word embeddings in TSV format')
    train_cnn_model_parser.add_argument('--n-labels', type=positive_int, required=True,
                                        help='Number of unique labels in the dataset')
    train_cnn_model_parser.add_argument('--val-data-path', type=file_path, default=None, nargs='+',
                                        action=UnnestSingletonListElement,
                                        help='Path to file containing the validation data in FastText format.'
                                             ' No validation will be performed during training if not specified.')
    train_cnn_model_parser.add_argument('--output-dir', type=dir_path, default='.',
                                        help='Path to directory in which to save the trained model')
    train_cnn_model_parser.add_argument('--batch-size', type=positive_int, default=32,
                                        help='Batch size to use during training')
    train_cnn_model_parser.add_argument('--n-epochs', type=positive_int, default=3,
                                        help='Number of epochs taken during training')
    train_cnn_model_parser.add_argument('--max-filter-s', type=positive_int, default=4,
                                        help='Maximum filter bank height')
    train_cnn_model_parser.add_argument('--min-filter-s', type=positive_int, default=2,
                                        help='Minimum filter bank height')
    train_cnn_model_parser.add_argument('--filter-s-step', type=positive_int, default=1,
                                        help='Step in filter bank size')
    train_cnn_model_parser.add_argument('--n-filter-channels', type=positive_int, default=2,
                                        help='Number of channels in filter bank')
    train_cnn_model_parser.add_argument('--hidden-size', type=positive_int, default=32,
                                        help='Size of hidden layers in the classifier')
    train_cnn_model_parser.add_argument('--eval-every-steps', type=positive_int, default=100,
                                        help='Perform evaluation on validation data every specified number of steps')


def _add_subparser_for_get_train_test_split(subparsers):
    train_test_split_parser = subparsers.add_parser(Tasks.TRAIN_TEST_SPLIT.value)
    train_test_split_parser.add_argument('--data-path', type=file_path, required=True,
                                         help='Path to file containing the data in fastText format')
    train_test_split_parser.add_argument('--train-size', type=proportion_float, default=0.8,
                                         help='Proportion of the dataset to include in the train split')
    train_test_split_parser.add_argument('--no-stratify', action='store_true',
                                         help='Do not split the data in a stratified fashion')
    train_test_split_parser.add_argument('--output-dir', type=dir_path, default='.',
                                         help='Path to directory in which to save the resulting files containing the'
                                              ' training and test data')


def _add_subparser_for_evaluate_embeddings_model(subparsers):
    evaluate_embeddings_model_parser = subparsers.add_parser(Tasks.EVALUATE_EMBEDDINGS_MODEL.value)
    evaluate_embeddings_model_parser.add_argument('--method', type=str, nargs='+', required=True,
                                                  action=UnnestSingletonListElement,
                                                  choices=[v.value for v in EntityEmbeddingMethod],
                                                  help='Entity embedding method to evaluate_embeddings_model')
    evaluate_embeddings_model_parser.add_argument('--train-data-path', type=file_path, nargs='+', required=True,
                                                  action=UnnestSingletonListElement,
                                                  help='Path to file containing the training data in fastText format'
                                                       ' (for training internal classifiers)')
    evaluate_embeddings_model_parser.add_argument('--test-data-path', type=file_path, nargs='+', required=True,
                                                  action=UnnestSingletonListElement,
                                                  help='Path to file containing the test data in fastText format')
    evaluate_embeddings_model_parser.add_argument('--validation-size', type=proportion_float, default=0.3,
                                                  help='Proportion of the dataset to use for hyperparameter tuning')
    evaluate_embeddings_model_parser.add_argument('--no-stratify', action='store_true',
                                                  help='Do not split the data in a stratified fashion')
    evaluate_embeddings_model_parser.add_argument('--param-grid-path', type=file_path,
                                                  help='Path to parameter grid in JSON format')
    evaluate_embeddings_model_parser.add_argument('--cv', type=int, default=5,
                                                  help='Number of folds to use when doing cross-validation')
    evaluate_embeddings_model_parser.add_argument('--embeddings-path', type=file_path,
                                                  help='Path to stored feature embeddings')
    evaluate_embeddings_model_parser.add_argument('--doc2vec-model-path', type=file_path,
                                                  help='Path to stored Doc2Vec model')
    evaluate_embeddings_model_parser.add_argument('--binary', action='store_true',
                                                  help='Embeddings are stored in binary format')
    evaluate_embeddings_model_parser.add_argument('--starspace-path', type=file_path,
                                                  help='Path to StarSpace executable')
    evaluate_embeddings_model_parser.add_argument('--results-path', type=dir_path, default='.',
                                                  help='Path to directory in which to save the results')
    evaluate_embeddings_model_parser.add_argument('--internal-clf', type=str,
                                                  choices=[e.value for e in InternalClassifier],
                                                  default=InternalClassifier.RANDOM_FOREST.value,
                                                  help='Internal classifier to use (if applicable)')
    evaluate_embeddings_model_parser.add_argument('--internal-clf-args', type=str, default='',
                                                  help='Arguments passed to internal classifier if applicable'
                                                       ' (key-value pairs such as val=1 enclose in quotes with no'
                                                       ' commas separated by spaces)')


def _add_subparser_for_evaluate_cnn_model(subparsers):
    evaluate_cnn_model_parser = subparsers.add_parser(Tasks.EVALUATE_CNN_MODEL.value)
    evaluate_cnn_model_parser.add_argument('--model-path', type=file_path, required=True,
                                           help='Path to trained model to evaluate')
    evaluate_cnn_model_parser.add_argument('--test-data-path', type=file_path, required=True,
                                           help='Path to file containing the test data in fastText format')
    evaluate_cnn_model_parser.add_argument('--unique-labels', type=int, nargs='+', required=True,
                                           help='Unique labels present in the dataset')
    evaluate_cnn_model_parser.add_argument('--class-names', type=str, nargs='+', required=True,
                                           help='Names associated with the labels (in same order)')
    evaluate_cnn_model_parser.add_argument('--batch-size', type=positive_int, default=32,
                                           help='Batch size to use during evaluation')
    evaluate_cnn_model_parser.add_argument('--results-path', type=dir_path, default='.',
                                           help='Path to directory in which to save the results')


def _task_get_entity_embeddings(parsed_args: dict):
    if parsed_args['method'] == EntityEmbeddingMethod.STARSPACE.value:
        # STARSPACE
        logger.info('Using StarSpace method.')
        if parsed_args['starspace_path'] is None:
            raise ValueError('path to StarSpace executable must be defined when using StarSpace')
        res_path = get_starspace_embeddings(parsed_args['starspace_path'], parsed_args['train_data_path'],
                                            parsed_args['output_dir'], parsed_args['starspace_args'])
    elif parsed_args['method'] == EntityEmbeddingMethod.WORD2VEC.value:
        # WORD2VEC
        logger.info('Using Word2Vec method.')
        res_path = get_word2vec_embeddings(parsed_args['train_data_path'], parsed_args['output_dir'],
                                           parsed_args['word2vec_args'])
    elif parsed_args['method'] == EntityEmbeddingMethod.FASTTEXT.value:
        # FASTTEXT
        logger.info('Using fastText method.')
        res_path = get_fasttext_embeddings(parsed_args['train_data_path'], parsed_args['output_dir'],
                                           parsed_args['fasttext_args'])
    elif parsed_args['method'] == EntityEmbeddingMethod.DOC2VEC.value:
        # DOC2VEC
        logger.info('Using Doc2Vec method.')
        res_path = get_doc2vec_embeddings(parsed_args['train_data_path'], parsed_args['output_dir'],
                                          parsed_args['doc2vec_args'])
    else:
        raise NotImplementedError('Method {0} not implemented'.format(parsed_args['method']))

    logger.info('Embeddings saved to {}'.format(res_path))


def _task_train_cnn_model(parsed_args: dict):
    train_cnn_model(
        train_data_path=parsed_args['train_data_path'],
        word_embeddings_path=parsed_args['word_embeddings_path'],
        n_labels=parsed_args['n_labels'],
        val_data_path=parsed_args['val_data_path'],
        output_dir=parsed_args['output_dir'],
        batch_size=parsed_args['batch_size'],
        n_epochs=parsed_args['n_epochs'],
        max_filter_s=parsed_args['max_filter_s'],
        min_filter_s=parsed_args['min_filter_s'],
        filter_s_step=parsed_args['filter_s_step'],
        n_filter_channels=parsed_args['n_filter_channels'],
        hidden_size=parsed_args['hidden_size'],
        eval_every_steps=parsed_args['eval_every_steps']
    )


def _task_train_test_split(parsed_args: dict):
    get_train_test_split(parsed_args['data_path'], parsed_args['output_dir'], parsed_args['train_size'],
                         not parsed_args['no_stratify'])


def _task_evaluate_embeddings_model(parsed_args: dict):
    logger.info('Performing evaluation of method \'{0}\''.format(parsed_args['method']))

    if parsed_args['embeddings_path'] is not None or (
            parsed_args['doc2vec_model_path'] is not None and parsed_args['method'] == 'doc2vec'):
        clf = _get_clf_stored_embeddings(parsed_args)
    else:
        clf = _get_clf_gs(parsed_args)

    evaluate_embeddings_model(clf, parsed_args['method'], parsed_args['test_data_path'], parsed_args['results_path'])


def _task_evaluate_cnn_model(parsed_args: dict):
    model = torch.load(parsed_args['model_path'], map_location=torch_device)

    test_dataset = FastTextFormatDataset(parsed_args['test_data_path'])
    test_data_loader = DataLoader(test_dataset, shuffle=True, batch_size=parsed_args['batch_size'],
                                  collate_fn=FastTextFormatDataset.collate)

    evaluate_cnn_model(
        model,
        test_data_loader=test_data_loader,
        results_path=parsed_args['results_path'],
        unique_labels=parsed_args['unique_labels'],
        class_names=parsed_args['class_names']
    )


def _get_clf_stored_embeddings(parsed_args: dict) -> AClassifier:
    """Get AClassifier instance initialized with stored embeddings."""

    logger.info('Initializing classifier using stored embeddings.')

    if parsed_args['method'] == EntityEmbeddingMethod.STARSPACE.value:
        # STARSPACE
        word_to_embedding = get_word_to_embedding(parsed_args['embeddings_path'], binary=parsed_args['binary'])
        clf = get_clf_starspace(word_to_embedding)
    elif parsed_args['method'] == EntityEmbeddingMethod.WORD2VEC.value or \
            parsed_args['method'] == EntityEmbeddingMethod.FASTTEXT.value or \
            parsed_args['method'] == EntityEmbeddingMethod.PRE_TRAINED_FROM_FILE.value:
        # STORED EMBEDDINGS
        word_to_embedding = get_word_to_embedding(parsed_args['embeddings_path'], binary=parsed_args['binary'])
        clf_internal = _get_internal_clf(parsed_args['internal_clf'])
        clf = get_clf_with_internal_clf(word_to_embedding, parsed_args['train_data_path'], clf_internal,
                                        parsed_args['internal_clf_args'])
    elif parsed_args['method'] == EntityEmbeddingMethod.DOC2VEC.value:
        # DOC2VEC
        clf_internal = _get_internal_clf(parsed_args['internal_clf'])
        doc2vec_model = Doc2Vec.load(parsed_args['doc2vec_model_path'])
        clf = get_clf_with_internal_clf_doc2vec(doc2vec_model, parsed_args['train_data_path'], clf_internal,
                                                parsed_args['internal_clf_args'])
    else:
        raise NotImplementedError('Method \'{0}\' not implemented.'.format(parsed_args['method']))
    return clf


def _get_clf_gs(parsed_args: dict) -> AClassifier:
    """Get AClassifier instance initialized with embeddings computed using parameters obtained with a grid-search."""

    logger.info('Training classifier using grid-search.')

    # assert embedding method supported
    valid_methods = [e.value for e in EntityEmbeddingMethod if e != EntityEmbeddingMethod.STARSPACE]
    method = parsed_args['method']
    if not all([el in valid_methods for el in ([method] if isinstance(method, str) else method[0])]):
        raise NotImplementedError('Method \'{}\' not supported when using grid search.'.format(method))

    train_data_path = parsed_args['train_data_path']

    train_paths, val_paths = get_train_and_val_paths_for_multiple_train_files(
        lambda path, train_suffix, test_suffix: get_train_test_split(
            path,
            '.',
            parsed_args['validation_size'],
            not parsed_args['no_stratify'],
            train_suffix=train_suffix,
            test_suffix=test_suffix
        ),
        [train_data_path] if isinstance(train_data_path, str) else train_data_path,
        'gs_train',
        'gs_val'
    )
    param_grid = parse_param_grid(parsed_args['param_grid_path'])
    clf_internal = _get_internal_clf(parsed_args['internal_clf'])

    clf = get_clf_with_internal_clf_gs(
        train_data_path=train_paths if len(train_paths) > 1 else train_paths[0],
        validation_data_path=val_paths if len(val_paths) > 1 else val_paths[0],
        param_grid=param_grid,
        embedding_method=method,
        clf_internal=clf_internal,
        cv=parsed_args['cv'],
        embeddings_path=parsed_args['embeddings_path'],
        binary=parsed_args['binary'],
        starspace_path=parsed_args['starspace_path']
    )

    return clf


def _get_internal_clf(internal_clf_kind: str):
    if internal_clf_kind == InternalClassifier.LOGISTIC_REGRESSION.value:
        return LogisticRegression
    elif internal_clf_kind == InternalClassifier.RANDOM_FOREST.value:
        return RandomForestClassifier
    elif internal_clf_kind == InternalClassifier.SVC.value:
        return SVC
    elif internal_clf_kind == InternalClassifier.DUMMY.value:
        return DummyClassifier
    else:
        raise NotImplementedError('Classifier {0} not implemented.'.format(internal_clf_kind))


class UnnestSingletonListElement(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values[0] if len(values) == 1 else values)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
