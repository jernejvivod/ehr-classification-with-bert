from typing import List, Union, Iterator

from gensim.models import FastText

from classification_with_embeddings.evaluation.embedders.a_doc_embedder import ADocEmbedder


class FastTextDocEmbedder(ADocEmbedder):
    def get_word_to_embedding(self, train_sentences: Union[List[List[str]], Iterator], y: list):
        ft_model = FastText(**self.embedding_kwargs)
        ft_model.build_vocab(corpus_iterable=train_sentences)
        ft_model.train(corpus_iterable=train_sentences, total_examples=len(train_sentences), epochs=10)
        return {k: ft_model.wv[k] for k in ft_model.wv.index_to_key}
