from typing import Tuple, List, Iterable

from classification_with_embeddings import LABEL_WORD_PREFIX


def _fasttext_data_to_x_y(data_path: str) -> Tuple[List[List[str]], List[str]]:
    """Transform data in FastText format to a List[List[str]] (list of tokenized sentences) and List[str] (labels).

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
            y.append(gt_label)

    return x, y


def _fasttext_data_to_x_y_multiple(data_paths: Iterable[str]) -> Tuple[List[List[List[str]]], List[str]]:
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