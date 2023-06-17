from typing import Dict, List

import torch
import torch.nn as nn

from classification_with_embeddings.embedding.cnn.model.cnn_feature_extractor_model import CnnTextFeatureExtractionModel


class CnnTextClassificationModel(nn.Module):
    def __init__(
            self,
            word_to_embedding: Dict[str, torch.tensor],
            n_classes: int,
            max_filter_s: int = 4,
            min_filter_s: int = 2,
            filter_s_step: int = 1,
            n_filter_channels=2,
            hidden_size=32
    ):
        """CNN-based document classification model. The model slides filters from filter banks of increasing heights.
        The maximum value from each feature map of each filter bank is taken, and the values are concatenated into a
        feature vector. Dense layers are then used to perform the document classification.

        :param word_to_embedding: mapping of words to their embeddings
        :param max_filter_s: maximum filter bank height
        :param min_filter_s: minimum filter bank height
        :param filter_s_step: step in filter bank size
        :param n_filter_channels: number of channels in filter bank
        :param hidden_size: size of the hidden layers in the classifier
        """
        super().__init__()

        self.feature_extractor = CnnTextFeatureExtractionModel(
            word_to_embedding=word_to_embedding,
            max_filter_s=max_filter_s,
            min_filter_s=min_filter_s,
            filter_s_step=filter_s_step,
            n_filter_channels=n_filter_channels,
        )

        self.relu = nn.ReLU(inplace=True)

        # initialize classifier
        n_filter_banks = ((max_filter_s - min_filter_s) + 1) // filter_s_step
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(n_filter_channels * n_filter_banks, hidden_size),
            self.relu,

            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, hidden_size),
            self.relu,

            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, x: List[List[str]]):
        # get feature vector
        feature_vector = self.feature_extractor(x)

        # perform classification
        return self.classifier(feature_vector)
