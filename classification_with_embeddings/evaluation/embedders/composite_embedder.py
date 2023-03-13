from typing import List

import numpy as np

from classification_with_embeddings.evaluation.embedders.a_doc_embedder import ADocEmbedder


class CompositeEmbedder(ADocEmbedder):
    def __init__(self, embedders: List[ADocEmbedder], vector_size: int = 100):
        super().__init__(vector_size=vector_size)
        self.embedders = embedders

    def get_word_to_embedding(self, train_sentences: List[List[str]]):
        raise NotImplementedError('Method not supported.')

    def fit(self, train_sentences: List[List[List[str]]], y):
        for train_sentences_sect, embedder in zip(CompositeEmbedder._transform_sentence_lists_to_list_of_lists_of_sentences(train_sentences), self.embedders):
            embedder.fit(train_sentences_sect, y)
        return self

    def transform(self, sentences: List[List[List[str]]]):
        embeddings = []
        for sentences_sect, embedder in zip(CompositeEmbedder._transform_sentence_lists_to_list_of_lists_of_sentences(sentences), self.embedders):
            embeddings.append(embedder.transform(sentences_sect))

        return np.hstack(embeddings)

    @staticmethod
    def _transform_sentence_lists_to_list_of_lists_of_sentences(sentences: List[List[List[str]]]) -> List[List[List[str]]]:
        """Transform data to lists of:
        [[[tokens for sentence_11], [tokens for sentence_21], ..., [tokens for sentence_m1]], [[tokens for sentence_12], [tokens for sentence_22], ..., [tokens for sentence_m2]], ... , [[tokens for sentence_1n], [tokens for sentence_2n], ..., [tokens for sentence_mn]]]

        from

        [[[tokens for sentence_11], [tokens for sentence_12], ..., [tokens for sentence_1n]], [[tokens for sentence_21], [tokens for sentence_22], ..., [tokens for sentence_2n]], ..., [[tokens for sentence_m1], [tokens for sentence_m2], ..., [tokens for sentence_mn]]]

        Where sentence_ij is the j-th sentence for i-th entity.
        """
        return [[sections[idx] for sections in [s for s in sentences]] for idx in range(len(sentences[0]))]
