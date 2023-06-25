from typing import Union, List, Iterator

from gensim.models import Word2Vec

from classification_with_embeddings.evaluation.embedders.a_doc_embedder import ADocEmbedder


class Word2VecDocEmbedder(ADocEmbedder):
    def get_word_to_embedding(self, train_sentences: Union[List[List[str]], Iterator], y: list):
        word2vec_model = Word2Vec(
            sentences=train_sentences,
            vector_size=self.vector_size,
            min_count=1,
            **self.method_kwargs
        )

        return {k: word2vec_model.wv[k] for k in word2vec_model.wv.index_to_key}
