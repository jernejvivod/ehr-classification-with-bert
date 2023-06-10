import argparse
import os
from typing import Union, Optional

from datasets import load_dataset, Split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ehr_classification_with_bert import logger


def get_dataloader(dataset_name: str,
                   split: Union[str, Split] = Split.ALL,
                   batch_size: int = 16,
                   truncate_dataset_to: Optional[int] = None) -> DataLoader:
    """Get PyTorch DataLoader for specified dataset.

    :param dataset_name: Name fo dataset to use
    :param split: Part of dataset to use (train, validation, test, all)
    :param batch_size: Batch size to use
    :param truncate_dataset_to: If not None, truncate the dataset to have the specified number of samples
    """

    logger.info('Obtaining DataLoader for dataset: %s, split: %s, batch_size: %d', dataset_name, split, batch_size)

    # load dataset
    dataset = load_dataset(dataset_name, split=split)

    # tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    tokenized_datasets = dataset.map(
        lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True), batched=True
    )

    # preprocess
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    tokenized_datasets = tokenized_datasets.rename_column('label', 'labels')
    tokenized_datasets.set_format('torch')

    # shuffle and optionally truncate the dataset
    processed_dataset = tokenized_datasets.shuffle(seed=42)
    if truncate_dataset_to:
        processed_dataset = processed_dataset.select(range(truncate_dataset_to))

    return DataLoader(processed_dataset, shuffle=True, batch_size=batch_size)


def get_relative_path(file_path: str, path: str) -> str:
    return os.path.join(os.path.dirname(file_path), path)


def argparse_type_positive_int(value):
    try:
        value = int(value)
        if value <= 0:
            raise argparse.ArgumentTypeError("{} is not a positive integer.".format(value))
        return value
    except ValueError:
        raise argparse.ArgumentTypeError("{} is not an integer.".format(value))
