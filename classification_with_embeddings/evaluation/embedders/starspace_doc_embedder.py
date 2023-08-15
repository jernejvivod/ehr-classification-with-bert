import os
import subprocess
from typing import Union, List, Iterator, Dict, Optional

import numpy as np

from classification_with_embeddings import LABEL_WORD_PREFIX
from classification_with_embeddings.embedding import embed_util
from classification_with_embeddings.evaluation.embedders.a_doc_embedder import ADocEmbedder
from classification_with_embeddings.util.errors import EmbeddingError


class StarspaceDocEmbedder(ADocEmbedder):
    _TMP_TRAINING_DATA_PATH = './train_data_starspace.txt'
    _TMP_STARSPACE_MODEL_NAME = 'starspace_model'

    def __init__(self, embedding_kwargs: Optional[dict] = None, **model_init_kwargs):
        super().__init__(embedding_kwargs=embedding_kwargs, **model_init_kwargs)
        self.starspace_path = model_init_kwargs['starspace_path']

    def get_word_to_embedding(self, train_sentences: Union[List[List[str]], Iterator], y: list) -> Dict[str, np.ndarray]:

        # write training data to a temporary file
        self._write_training_data_to_file(train_sentences, y)

        # get StarSpace embeddings
        p = subprocess.run(
            [self.starspace_path, 'train', '-trainFile', self._TMP_TRAINING_DATA_PATH, '-model', './' + self._TMP_STARSPACE_MODEL_NAME]
        )

        if p.returncode != 0:
            raise EmbeddingError('StarSpace', p.returncode)

        word_to_embedding = embed_util.get_word_to_embedding('./starspace_model.tsv')

        self._remove_tmp_files()

        return word_to_embedding

    def _write_training_data_to_file(self, train_sentences: Union[List[List[str]], Iterator], y: list):
        # write training data to file
        with open(self._TMP_TRAINING_DATA_PATH, 'w') as f:
            for idx in range(len(train_sentences)):
                f.write(' '.join(train_sentences[idx]))
                f.write(' ' + LABEL_WORD_PREFIX + str(y[idx]))
                f.write('\n')

    def _remove_tmp_files(self):
        os.remove(self._TMP_TRAINING_DATA_PATH)
        os.remove(self._TMP_STARSPACE_MODEL_NAME)
        os.remove(self._TMP_STARSPACE_MODEL_NAME + '.tsv')
