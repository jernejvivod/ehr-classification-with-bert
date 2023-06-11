import argparse
import os
from typing import Optional, Union

import datasets
from datasets import TextClassification, ClassLabel, Split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def get_dataloader(data_file_path: str,
                   n_labels: int,
                   split: Union[str, Split],
                   batch_size: int = 16,
                   truncate_dataset_to: Optional[int] = None) -> DataLoader:
    """Get PyTorch DataLoader for specified dataset.

    :param data_file_path: Path to file containing the dataset
    :param n_labels: Number of unique labels in the dataset
    :param split: Part of dataset to use (train, validation, test, all)
    :param batch_size: Batch size to use
    :param truncate_dataset_to: If not None, truncate the dataset to have the specified number of samples
    """

    # load data from file
    dataset = datasets.load_dataset('text', data_files={split: data_file_path})[split]

    label_prefix_length = len('__label__')

    # preprocess the dataset
    def pre_process_dataset(example):
        return {
            'text': ' '.join(example['text'].split(' ')[:-1]),
            'label': int(example['text'].split(' ')[-1][label_prefix_length:])}

    dataset = dataset.map(pre_process_dataset)
    dataset = dataset.cast_column(
        'label',
        ClassLabel(num_classes=n_labels, names=['__label__{}'.format(i) for i in range(n_labels)])
    )
    dataset = dataset.prepare_for_task(TextClassification(text_column="text", label_column="label"))

    # tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True), batched=True
    )

    # remove redundant column and aset format
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    tokenized_datasets.set_format('torch')

    # shuffle and optionally truncate the dataset
    processed_dataset = tokenized_datasets.shuffle(seed=42)
    if truncate_dataset_to:
        processed_dataset = processed_dataset.select(range(truncate_dataset_to))

    return DataLoader(processed_dataset, shuffle=True, batch_size=batch_size)


def argparse_type_positive_int(value):
    try:
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError("{} is not a positive integer.".format(value))
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("{} is not an integer.".format(value))


def argparse_type_file_path(path: str) -> str:
    if os.path.exists(path):
        return path
    else:
        raise argparse.ArgumentTypeError('File \'{0}\' does not exist.'.format(path))


def argparse_type_dir_path(path: str) -> str:
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError('\'{0}\' is not a directory.'.format(path))
