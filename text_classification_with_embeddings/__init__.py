from enum import Enum


class Tasks(Enum):
    GET_ENTITY_EMBEDDINGS = 'get-entity-embeddings'
    EVALUATE = 'evaluate'


class EntityEmbeddingMethod(Enum):
    AGGREGATED_WORD2VEC = 'aggregated-word2vec'
    STARSPACE = 'starspace'
