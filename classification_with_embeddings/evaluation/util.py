import os
from typing import Tuple, List, Iterable

from classification_with_embeddings import LABEL_WORD_PREFIX
from classification_with_embeddings.evaluation import logger


def _fasttext_data_to_x_y(data_path: str) -> Tuple[List[List[str]], List[int]]:
    """Transform data in FastText format to a List[List[str]] (list of tokenized sentences) and List[int] (labels).

    :param data_path: path to data
    :return: tuple containing the tokenized sentences and labels
    """

    x = []
    y = []

    with open(data_path, 'r') as f:
        for idx, sample in enumerate(f):
            x.append([w for w in sample.split() if LABEL_WORD_PREFIX not in w])

            # find ground-truth label
            label_search = [el for el in sample.split() if LABEL_WORD_PREFIX in el]
            if len(label_search) == 0:
                raise ValueError('Label not found in sample {0} in {1}'.format(idx, data_path))
            gt_label = label_search[0].replace(LABEL_WORD_PREFIX, '')
            try:
                y.append(int(gt_label))
            except ValueError:
                raise ValueError('All labels in the provided dataset should be encoded as integer values.')

    return x, y


def _fasttext_data_to_x_y_multiple(data_paths: Iterable[str]) -> Tuple[List[List[List[str]]], List[int]]:
    """Transform multiple files relating to data in FastText format to a List[List[List[str]]] (list of lists of tokenized sentences all related to one entity).

    Transform data to lists of:
    [[[tokens for sentence_11], [tokens for sentence_12], ..., [tokens for sentence_1n]], [[tokens for sentence_21], [tokens for sentence_22], ..., [tokens for sentence_2n]], ..., [[tokens for sentence_m1], [tokens for sentence_m2], ..., [tokens for sentence_mn]]]

    from

    [[[tokens for sentence_11], [tokens for sentence_21], ..., [tokens for sentence_m1]], [[tokens for sentence_12], [tokens for sentence_22], ..., [tokens for sentence_m2]], ... , [[tokens for sentence_1n], [tokens for sentence_2n], ..., [tokens for sentence_mn]]]

    Where sentence_ij is the j-th sentence for i-th entity.

    This is needed for splitting the data (e.g. when doing cross-validation).

    :param data_paths: paths to data
    :return: tuple containing the tokenized lists of sentences and labels
    """
    xs = []
    ys = []

    for data_paths in data_paths:
        x, y = _fasttext_data_to_x_y(data_paths)
        xs.append(x)
        ys.append(y)

    xs_trans = [[sections[idx] for sections in [s for s in xs]] for idx in range(len(xs[0]))]
    return xs_trans, ys[0]


def _write_evaluation_prediction_data_to_file(method: str,
                                              results_path: str,
                                              y_pred: list = None,
                                              scores: list = None,
                                              y_true: list = None,
                                              labels: list = None):
    """Write classifier test prediction results to file.

    :param method: embedding method(s) used
    :param results_path: path to directory in which to store the results
    :param y_pred: predictions of the classifier
    :param scores: scores for classes (probabilities)
    :param y_true: ground truth values
    :param labels: unique labels
    """

    output_file_path = os.path.abspath(os.path.join(results_path, method + '_data' + '.txt'))
    logger.info('Saving prediction evaluation data to {0}'.format(output_file_path))

    with open(output_file_path, 'w') as f:
        f.write('method: {}\n'.format(method))
        if y_pred is not None:
            f.write('y_pred: [{}]\n'.format(','.join(str(el) for el in y_pred)))
        if scores is not None:
            f.write('scores: [{}]\n'.format(','.join(str(el) for el in scores)))
        if y_true is not None:
            f.write('y_true: [{}]\n'.format(','.join(str(el) for el in y_true)))
        if labels is not None:
            f.write('labels: [{}]\n'.format(','.join(str(el) for el in labels)))
