from gensim.models import Word2Vec

from classification_with_embeddings.evaluation.embedders.a_doc_embedder import ADocEmbedder


class Word2VecDocEmbedder(ADocEmbedder):
    def get_word_to_embedding(self, train_sentences):
        word2vec_model = Word2Vec(train_sentences, vector_size=self.vector_size, min_count=1)
        return {k: word2vec_model.wv[k] for k in word2vec_model.wv.index_to_key}
