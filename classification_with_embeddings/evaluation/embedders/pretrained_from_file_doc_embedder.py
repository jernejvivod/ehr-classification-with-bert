from classification_with_embeddings.embedding import embed_util

from classification_with_embeddings.evaluation.embedders.a_doc_embedder import ADocEmbedder


class PreTrainedFromFileDocEmbedder(ADocEmbedder):
    def __init__(self, path_to_embedding: str, binary: bool, **kwargs):
        super().__init__(**kwargs)
        self.path_to_embedding = path_to_embedding
        self.binary = binary
        self.word_to_embedding = None

    def get_word_to_embedding(self, _1, _2):
        if not self.word_to_embedding:
            self.word_to_embedding = embed_util.get_word_to_embedding(self.path_to_embedding, self.binary)
        return self.word_to_embedding
