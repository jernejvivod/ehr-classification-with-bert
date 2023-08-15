from typing import Optional

from classification_with_embeddings.embedding import embed_util
from classification_with_embeddings.evaluation.embedders.a_doc_embedder import ADocEmbedder


class PreTrainedFromFileDocEmbedder(ADocEmbedder):
    def __init__(self, embedding_kwargs: Optional[dict] = None, **model_init_kwargs):
        super().__init__(embedding_kwargs, **model_init_kwargs)
        self.embeddings_path = model_init_kwargs['embeddings_path']
        self.binary = model_init_kwargs['binary']
        self.word_to_embedding = None

    def get_word_to_embedding(self, _1, _2):
        if not self.word_to_embedding:
            self.word_to_embedding = embed_util.get_word_to_embedding(self.embeddings_path, self.binary)
        return self.word_to_embedding
