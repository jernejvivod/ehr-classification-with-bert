import argparse
import os
from typing import Optional, Union, Iterable, Any, Dict

import datasets
from datasets import TextClassification, ClassLabel, Split
from datasets.formatting.formatting import LazyRow
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def get_dataloader(data_file_path: str,
                   n_labels: int,
                   split: Union[str, Split],
                   batch_size: int = 16,
                   truncate_dataset_to: Optional[int] = None,
                   split_above_tokens_limit: bool = False,
                   group_splits: bool = False) -> DataLoader:
    """Get PyTorch DataLoader for specified dataset.

    :param data_file_path: Path to file containing the dataset
    :param n_labels: Number of unique labels in the dataset
    :param split: Part of dataset to use (train, validation, test, all)
    :param batch_size: Batch size to use
    :param truncate_dataset_to: If not None, truncate the dataset to have the specified number of samples
    :param split_above_tokens_limit: If True, split examples above the tokenization length limit into multiple examples
    :param group_splits: if True, group splits of examples into lists
    """

    # load data from file
    dataset = datasets.load_dataset('text', data_files={split: data_file_path})[split]

    label_prefix_length = len('__label__')

    # preprocess the dataset
    def pre_process_dataset(example):
        return {
            'text': ' '.join(example['text'].split(' ')[:-1]),
            'label': int(example['text'].split(' ')[-1][label_prefix_length:])
        }

    dataset = dataset.map(pre_process_dataset, load_from_cache_file=False)
    dataset = dataset.cast_column(
        'label',
        ClassLabel(num_classes=n_labels, names=['__label__{}'.format(i) for i in range(n_labels)])
    )
    dataset = dataset.prepare_for_task(TextClassification(text_column="text", label_column="label"))

    # tokenization padding and truncation depend on whether we are splitting examples with length above the limit
    tokenization_padding = 'max_length' if not split_above_tokens_limit else 'do_not_pad'
    tokenization_truncation = not split_above_tokens_limit

    # tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    tokenized_datasets = dataset.map(
        lambda example: tokenizer(example['text'], padding=tokenization_padding, truncation=tokenization_truncation),
        batched=True
    )

    # if splitting examples above the number of tokens limit, split
    if split_above_tokens_limit:
        tokenized_datasets = tokenized_datasets.map(
            lambda example: split_example_above_length_limit(
                example,
                ['input_ids', 'token_type_ids', 'attention_mask'],
                tokenizer.model_max_length,
                group_splits
            ),
            batched=True,
            batch_size=1,  # must be equal to 1
            remove_columns=tokenized_datasets.column_names,
            load_from_cache_file=False
        )

    # remove redundant column and set format
    tokenized_datasets.set_format('torch')

    # shuffle and optionally truncate the dataset
    processed_dataset = tokenized_datasets.shuffle(seed=42)
    if truncate_dataset_to:
        processed_dataset = processed_dataset.select(range(truncate_dataset_to))

    return DataLoader(processed_dataset, shuffle=True, batch_size=batch_size, )


def split_example_above_length_limit(
        example: LazyRow,
        columns_to_split: Iterable[str],
        max_tokens: int,
        group_splits: bool) -> Dict[str, Any]:
    """Split tokenized examples with more tokens than the specified maximum number.
    This function is meant to be passed to the 'map' method used to transform a Hugging Face dataset.

    Make sure to set the 'batched' parameter values of the 'map' method to 'True'
    and to set the batch_size parameter value equal to 1.

    :param example: Example to process
    :param columns_to_split: Names of columns to split into multiple examples
    :param max_tokens: Maximum number of tokens
    :param group_splits: if True, group splits of examples into lists
    """

    example_with_split_columns = {col: [] for col in columns_to_split}

    # go over columns that should be split
    for col in columns_to_split:

        # number of complete groups
        n_complete_groups = len(example[col][0]) // max_tokens

        # size of incomplete group
        incomplete_group_l = len(example[col][0]) % max_tokens

        # get split get splits for column
        example_with_split_columns[col] += \
            [example[col][0][max_tokens * k: max_tokens * (k + 1)] for k in range(n_complete_groups)]

        # if incomplete group has elements, pad and add
        if incomplete_group_l != 0:
            example_with_split_columns[col] += [example[col][0][n_complete_groups * max_tokens:]]
            example_with_split_columns[col][-1] += [0] * (max_tokens - len(example_with_split_columns[col][-1]))

    # number of segments the example was split into
    n_segments = len(example_with_split_columns[next(iter(example_with_split_columns.keys()))])

    # repeat other columns (that were not split)
    repeated_other_columns = {
        col: [example[col][0] for _ in range(n_segments)] for col in example.keys() if col not in columns_to_split
    }

    # if grouping splits, enclose examples comprising the split example in a list
    if group_splits:
        example_with_split_columns = {k: [v] for k, v in example_with_split_columns.items()}
        repeated_other_columns = {k: [v] for k, v in repeated_other_columns.items()}

    return {**example_with_split_columns, **repeated_other_columns}


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
