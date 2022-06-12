import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from classification_with_embeddings import Tasks, EntityEmbeddingMethod, InternalClassifier
from classification_with_embeddings import logger
from classification_with_embeddings.embedding import embed
from classification_with_embeddings.embedding.embed import get_word_to_embedding
from classification_with_embeddings.evaluation.evaluate import evaluate
from classification_with_embeddings.evaluation.get_clf import get_clf_with_internal_clf, get_clf_starspace
from classification_with_embeddings.evaluation.train_test_split import get_train_test_split
from classification_with_embeddings.util.argparse import file_path, dir_path, proportion_float


def main(**kwargs):
    if kwargs['task'] == Tasks.GET_ENTITY_EMBEDDINGS.value:
        # COMPUTING DOCUMENT EMBEDDINGS FROM fastText FORMAT INPUT
        logger.info('Obtaining entity embeddings')
        task_get_entity_embeddings(**kwargs)
    elif kwargs['task'] == Tasks.TRAIN_TEST_SPLIT.value:
        # OBTAINING FILES CORRESPONDING TO A TRAIN-TEST SPLIT
        logger.info('Performing train-test split')
        task_train_test_split(**kwargs)
    elif kwargs['task'] == Tasks.EVALUATE.value:
        # EVALUATING EMBEDDING-BASED CLASSIFIERS
        logger.info('Performing evaluation')
        task_evaluate(**kwargs)
    else:
        raise NotImplementedError('Task {0} not implemented'.format(kwargs['task']))


def task_get_entity_embeddings(**kwargs):
    if kwargs['method'] == EntityEmbeddingMethod.STARSPACE.value:
        # STARSPACE
        logger.info('Using StarSpace method')
        if kwargs['starspace_path'] is None:
            raise ValueError('path to StarSpace executable must be defined when using StarSpace')
        embed.get_starspace_entity_embeddings(kwargs['starspace_path'], kwargs['train_data_path'], kwargs['output_dir_path'], kwargs['starspace_args'])
    elif kwargs['method'] == EntityEmbeddingMethod.WORD2VEC.value:
        # WORD2VEC
        logger.info('Using Word2Vec method')
        embed.get_word2vec_embeddings(kwargs['train_data_path'], kwargs['output_dir_path'], kwargs['word2vec_args'])
    elif kwargs['method'] == EntityEmbeddingMethod.FASTTEXT.value:
        # FASTTEXT
        logger.info('Using fastText method')
        embed.get_fasttext_embeddings(kwargs['train_data_path'], kwargs['output_dir_path'], kwargs['fasttext_args'])
    else:
        raise NotImplementedError('Method {0} not implemented'.format(kwargs['method']))


def task_train_test_split(**kwargs):
    get_train_test_split(kwargs['data_path'], kwargs['output_dir'], kwargs['train_size'], not kwargs['no_stratify'])


def task_evaluate(**kwargs):
    word_to_embedding = get_word_to_embedding(kwargs['embeddings_path'])
    if kwargs['method'] == EntityEmbeddingMethod.STARSPACE.value:
        # STARSPACE
        logger.info('Evaluating Starspace method')
        clf = get_clf_starspace(word_to_embedding)
    elif kwargs['method'] == EntityEmbeddingMethod.WORD2VEC.value or kwargs['method'] == EntityEmbeddingMethod.FASTTEXT.value:
        # WORD2VEC or FASTTEXT
        logger.info('Evaluating {0} method'.format('Word2Vec' if kwargs['method'] == EntityEmbeddingMethod.WORD2VEC.value else 'fastText'))

        # set internal classifier
        if kwargs['internal_clf'] == InternalClassifier.LOGISTIC_REGRESSION.value:
            clf_internal = LogisticRegression
        elif kwargs['internal_clf'] == InternalClassifier.RANDOM_FOREST.value:
            clf_internal = RandomForestClassifier
        elif kwargs['internal_clf'] == InternalClassifier.SVC.value:
            clf_internal = SVC
        else:
            raise NotImplementedError('Classifier {0} not implemented'.format(kwargs['internal_clf']))
        clf = get_clf_with_internal_clf(word_to_embedding, kwargs['train_data_path'], clf_internal, kwargs['internal_clf_args'])
    else:
        raise NotImplementedError('Method {0} not implemented'.format(kwargs['method']))
    evaluate(clf, kwargs['method'], kwargs['test_data_path'], kwargs['results_path'])


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(prog='classification-with-embeddings')
    subparsers = parser.add_subparsers(required=True, dest='task', help='Task to run')

    # COMPUTING DOCUMENT EMBEDDINGS FROM fastText FORMAT INPUT
    get_entity_embeddings_parser = subparsers.add_parser(Tasks.GET_ENTITY_EMBEDDINGS.value)
    get_entity_embeddings_parser.add_argument('--method', type=str, default=EntityEmbeddingMethod.STARSPACE.value,
                                              choices=[v.value for v in EntityEmbeddingMethod], help='Method of generating entity embeddings')
    get_entity_embeddings_parser.add_argument('--train-data-path', type=file_path, required=True, help='Path to file containing the training data in fastText format')
    get_entity_embeddings_parser.add_argument('--output-dir-path', type=dir_path, default='.', help='Path to directory in which to save the embeddings')
    get_entity_embeddings_parser.add_argument('--starspace-path', type=file_path, help='Path to StarSpace executable')
    get_entity_embeddings_parser.add_argument('--starspace-args', type=str, default='',
                                              help='Arguments passed to StarSpace implementation (key-value pairs such as val=1 enclose in quotes with no commas separated by spaces)')
    get_entity_embeddings_parser.add_argument('--word2vec-args', type=str, default='',
                                              help='Arguments passed to Word2Vec implementation (key-value pairs such as val=1 enclose in quotes with no commas separated by spaces)')
    get_entity_embeddings_parser.add_argument('--fasttext-args', type=str, default='',
                                              help='Arguments passed to fastText implementation (key-value pairs such as val=1 enclose in quotes with no commas separated by spaces)')

    # DATA TRAIN-TEST SPLIT
    train_test_split_parser = subparsers.add_parser(Tasks.TRAIN_TEST_SPLIT.value)
    train_test_split_parser.add_argument('--data-path', type=file_path, required=True, help='Path to file containing the data in fastText format')
    train_test_split_parser.add_argument('--train-size', type=proportion_float, default=0.8, help='Proportion of the dataset to include in the train split')
    train_test_split_parser.add_argument('--no-stratify', action='store_true', help='Do not split the data in a stratified fashion')
    train_test_split_parser.add_argument('--output-dir', type=dir_path, default='.', help='Path to directory in which to save the resulting files containing the training and test data')

    # DOCUMENT EMBEDDING EVALUATION
    evaluate_parser = subparsers.add_parser(Tasks.EVALUATE.value)
    evaluate_parser.add_argument('--method', type=str, required=True,
                                 choices=[v.value for v in EntityEmbeddingMethod], help='Entity embedding method to evaluate')
    evaluate_parser.add_argument('--train-data-path', type=file_path, required=True, help='Path to file containing the training data in fastText format (for training internal classifiers)')
    evaluate_parser.add_argument('--test-data-path', type=file_path, required=True, help='Path to file containing the test data in fastText format')
    evaluate_parser.add_argument('--embeddings-path', type=file_path, required=True, help='Path to stored feature embeddings')
    evaluate_parser.add_argument('--results-path', type=dir_path, default='.', help='Path to directory in which to save the results')
    evaluate_parser.add_argument('--internal-clf', type=str, default=InternalClassifier.LOGISTIC_REGRESSION.value, help='Internal classifier to use (if applicable)')
    evaluate_parser.add_argument('--internal-clf-args', type=str, default='',
                                 help='Arguments passed to internal classifier if applicable (key-value pairs such as val=1 enclose in quotes with no commas separated by spaces)')

    args = parser.parse_args()
    main(**vars(args))
