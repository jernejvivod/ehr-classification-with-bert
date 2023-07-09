from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as functional


class CnnTextFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            word_to_embedding: Dict[str, torch.tensor],
            max_filter_s: int = 4,
            min_filter_s: int = 2,
            filter_s_step: int = 1,
            n_filter_channels=2,
    ):
        """CNN-based document feature extraction model. The model slides filters from filter banks of increasing
        heights. The maximum value from each feature map of each filter bank is taken, and the values are concatenated
        into a feature vector.

        :param word_to_embedding: mapping of words to their embeddings
        :param max_filter_s: maximum filter bank height
        :param min_filter_s: minimum filter bank height
        :param filter_s_step: step in filter bank size
        :param n_filter_channels: number of channels in filter bank
        """
        super().__init__()

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self._word_to_embedding = word_to_embedding
        self._embedding_l = len(self._word_to_embedding[next(iter(self._word_to_embedding.keys()))])
        self._max_filter_s = max_filter_s

        # initialize filters
        self.filters = nn.ModuleList([nn.Conv2d(1, n_filter_channels, (s, self._embedding_l)) for s in
                                      range(min_filter_s, max_filter_s + 1, filter_s_step)])
        self.global_max_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

        # size of feature vector
        self.vector_size = len(self.filters) * n_filter_channels

    def _get_inputs_from_text(self, x: List[List[str]]) -> torch.tensor:
        """Get padded tensor for batch of text data (tokens).

        :param x: list of lists of text tokens (words) to use in the embeddings
        """

        # get matrix representation of documents
        emb_inputs = []
        length_longest_document = self._max_filter_s  # should pad to at least maximum filter bank height
        for example in x:
            emb_input_nxt = torch.stack(
                [self._word_to_embedding.get(w, torch.zeros(self._embedding_l)).to(self.device) for w in example])
            if emb_input_nxt.shape[0] > length_longest_document:
                length_longest_document = emb_input_nxt.shape[0]
            emb_inputs.append(emb_input_nxt)

        # pad matrices and concatenate into tensor
        padded_emb_inputs = []
        for emb_input in emb_inputs:
            n_padding_rows = length_longest_document - emb_input.shape[0]
            padded_emb_inputs.append(
                functional.pad(input=emb_input, pad=(0, 0, 0, n_padding_rows), mode='constant', value=0.0).unsqueeze(0)
            )

        return torch.stack(padded_emb_inputs)

    def forward(self, x: List[List[str]]):
        inputs = self._get_inputs_from_text(x).to(self.device)

        # get feature maps
        feature_maps = [self.relu(f(inputs)) for f in self.filters]

        # get vector of maximum values in each feature map of each filter bank
        one_max_vectors = [self.global_max_pooling(fv).squeeze(dim=(2, 3)) for fv in feature_maps]

        # concatenate the vectors of maximum feature map values into a feature vector
        return torch.cat(one_max_vectors, dim=1)

    def set_device(self, device):
        self.device = device
