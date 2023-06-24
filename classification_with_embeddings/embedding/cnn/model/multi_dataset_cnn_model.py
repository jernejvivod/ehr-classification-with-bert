from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from classification_with_embeddings.embedding.cnn.model.cnn_feature_extractor_model import CnnTextFeatureExtractionModel


class CompositeCnnTextClassificationModel(nn.Module):
    def __init__(
            self,
            n_datasets: int,
            word_to_embedding: Dict[str, torch.tensor],
            n_labels: int,
            max_filter_s: int = 4,
            min_filter_s: int = 2,
            filter_s_step: int = 1,
            n_filter_channels=2,
            hidden_size=32
    ):
        """CnnTextClassificationModel that supports multiple related datasets (the rows with the same indices correspond
        to the same entities). A feature vector is constructed for each document. The feature vectors are concatenated
        and Dense layers are then used to perform the classification.

        :param n_datasets: number of composite datasets that will be used
        :param word_to_embedding: mapping of words to their embeddings
        :param n_labels: number of unique labels
        :param max_filter_s: maximum filter bank height
        :param min_filter_s: minimum filter bank height
        :param filter_s_step: step in filter bank size
        :param n_filter_channels: number of channels in filter bank
        :param hidden_size: size of the hidden layers in the classifier
        """
        super().__init__()

        self.feature_extractors = nn.ModuleList([
            CnnTextFeatureExtractionModel(
                word_to_embedding=word_to_embedding,
                max_filter_s=max_filter_s,
                min_filter_s=min_filter_s,
                filter_s_step=filter_s_step,
                n_filter_channels=n_filter_channels,
            ) for _ in range(n_datasets)
        ])

        self.relu = nn.ReLU(inplace=True)

        # initialize classifier
        n_filter_banks = ((max_filter_s - min_filter_s) + 1) // filter_s_step
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(n_filter_channels * n_filter_banks * n_datasets, hidden_size),
            self.relu,

            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            self.relu,

            nn.Linear(hidden_size, n_labels)
        )

    def forward(self, x: Tuple[List[List[str]]]):
        # get feature vector
        feature_vector = torch.cat(
            [self.feature_extractors[i](list(map(lambda e: e[i], x))) for i in range(len(self.feature_extractors))],
            dim=1
        )

        # perform classification
        return self.classifier(feature_vector)

    def set_device(self, device):
        for fe in self.feature_extractors:
            fe.set_device(device)
