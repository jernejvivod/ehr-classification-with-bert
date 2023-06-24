from typing import Iterable

import torch
from torch.utils.data import Dataset

from classification_with_embeddings import LABEL_WORD_PREFIX


class FastTextFormatDataset(Dataset):
    def __init__(self, data_path: str):
        """Dataset for reading data in FastText format (space separated words followed by the label)

        :param data_path: Path to dataset
        """

        self.data_path = data_path
        self._length = None

    def __len__(self):
        if not self._length:
            with open(self.data_path, "rbU") as f:
                self._length = sum(1 for _ in f)

        return self._length

    def __getitem__(self, idx):
        with open(self.data_path, 'r') as f:
            for i, line in enumerate(f):
                if i == idx:
                    return self._get_example_and_label_from_fasttext_line(line)

    @staticmethod
    def _get_example_and_label_from_fasttext_line(line: str):
        split_line = line.split(' ')
        data = split_line[:-1]
        label = int(split_line[-1][len(LABEL_WORD_PREFIX):])

        return data, label

    # custom collation function to be used when initializing a DataLoader with this Dataset
    @staticmethod
    def collate(batch):
        data = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        return data, labels


class FastTextFormatCompositeDataset(Dataset):
    def __init__(self, data_paths: Iterable[str]):
        self.fasttext_format_datasets = [FastTextFormatDataset(data_path) for data_path in data_paths]
        self._length = None

    def __len__(self):
        if not self._length:
            self._length = len(self.fasttext_format_datasets[0])

        return self._length

    def __getitem__(self, idx):
        _, label = self.fasttext_format_datasets[0][idx]
        data = tuple(self.fasttext_format_datasets[i][idx][0] for i in range(len(self.fasttext_format_datasets)))
        return data, label

    # custom collation function to be used when initializing a DataLoader with this Dataset
    @staticmethod
    def collate(batch):
        data = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        return data, labels
