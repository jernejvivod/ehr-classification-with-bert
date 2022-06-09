from enum import Enum


class Tasks(Enum):
    GET_ENTITY_EMBEDDINGS = 'get-entity-embeddings'
    EVALUATE = 'evaluate'


class EntityEmbeddingMethod(Enum):
    WORD2VEC = 'word2vec'
    STARSPACE = 'starspace'
    FASTTEXT = 'fasttext'


LABEL_WORD_PREFIX = '__label__'
