import pathlib

import dask.dataframe as dd
import numpy as np
from sklearn import model_selection

from classification_with_embeddings.evaluation import logger
from classification_with_embeddings.util.files import delete_file


def get_train_test_split(data_path: str, output_dir: str, train_size: float = 0.8, stratify: bool = True) -> None:
    """Get files corresponding to a train-test split of the data in the specified file in fastText format.

    :param data_path: path to data containing the samples in fastText format
    :param output_dir: path to directory in which to save the resulting files containing the training and test data
    :param train_size: proportion of the dataset to include in the training split
    :param stratify: split the data in a stratified fashion
    """

    logger.info('Performing train-test split with with train_size={0}'.format(train_size))

    # parse data, extract labels and split
    samples = dd.read_csv(data_path).repartition(3)
    labels = samples.map_partitions(lambda d: d.iloc[:, 0].map(lambda x: [el for el in x.split() if '__label__' in el][0]), meta=('', str)).compute()
    y_train, y_test = model_selection.train_test_split(labels, train_size=train_size, stratify=labels if stratify else None, shuffle=True)

    # get paths to output files
    data_file_path = pathlib.Path(data_path)
    data_train_path = pathlib.Path(output_dir).joinpath(data_file_path.stem + '_train' + data_file_path.suffix)
    data_test_path = pathlib.Path(output_dir).joinpath(data_file_path.stem + '_test' + data_file_path.suffix)
    delete_file(data_train_path)
    delete_file(data_test_path)

    # write data
    for partition in samples.partitions:
        partition_c = partition.compute()
        index_partition_train = y_train.index[np.logical_and(y_train.index >= partition_c.index.start, y_train.index < partition_c.index.stop)]
        index_partition_test = y_test.index[np.logical_and(y_test.index >= partition_c.index.start, y_test.index < partition_c.index.stop)]
        train_batch_nxt = partition_c.loc[index_partition_train, :]
        test_batch_nxt = partition_c.loc[index_partition_test, :]
        train_batch_nxt.to_csv(data_train_path, mode='a', header=False, index=False)
        test_batch_nxt.to_csv(data_test_path, mode='a', header=False, index=False)
