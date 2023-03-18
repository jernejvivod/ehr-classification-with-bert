import pathlib
from typing import List, Tuple

from sklearn import model_selection

from classification_with_embeddings import LABEL_WORD_PREFIX
from classification_with_embeddings.evaluation import logger


def get_train_test_split(data_path: str, output_dir: str, train_size: float = 0.8, stratify: bool = True, train_suffix: str = 'train', test_suffix: str = 'test') -> Tuple[str, str]:
    """Get files corresponding to a train-test split of the data in the specified file in fastText format.

    :param data_path: path to data containing the samples in fastText format
    :param output_dir: path to directory in which to save the resulting files containing the training and test data
    :param train_size: proportion of the dataset to include in the training split
    :param stratify: split the data in a stratified fashion
    :param train_suffix: suffix to apply to file name containing the training data
    :param test_suffix: suffix to apply to file name containing the test data
    :return: tuple containing the path to the output training file and the path to the output test file
    """
    logger.info('Performing train-test split with with train_size={0}.'.format(train_size))

    labels = _read_labels(data_path)
    idxs_train, idxs_test = model_selection.train_test_split(range(len(labels)), train_size=train_size, stratify=labels if stratify else None, shuffle=True)
    return _write_train_and_test_data(data_path, output_dir, idxs_train, idxs_test, train_suffix, test_suffix)


def _read_labels(data_path: str):
    """Read labels from data in specified file"""

    labels = []
    with open(data_path, 'r') as f:
        for line in f:
            labels.append(line.split(' ')[-1][len(LABEL_WORD_PREFIX):])
    return labels


def _write_train_and_test_data(data_path: str, output_dir: str, idxs_train: List[int], idxs_test: List[int], train_suffix: str, test_suffix: str) -> Tuple[str, str]:
    """Write training and tests data to corresponding files and return the paths as a tuple"""

    data_file_path = pathlib.Path(data_path)
    data_train_path = pathlib.Path(output_dir).joinpath(data_file_path.stem + '_' + train_suffix + data_file_path.suffix)
    data_test_path = pathlib.Path(output_dir).joinpath(data_file_path.stem + '_' + test_suffix + data_file_path.suffix)

    with open(data_path, 'r') as f_data, open(data_train_path, 'w') as f_train, open(data_test_path, 'w') as f_test:
        for idx, line in enumerate(f_data):
            if idx in idxs_train:
                f_train.write(line)
            if idx in idxs_test:
                f_test.write(line)

    return str(data_train_path), str(data_test_path)
