import logging
from enum import Enum

# module logger

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Tasks(Enum):
    GET_ENTITY_EMBEDDINGS = 'get-entity-embeddings'
    TRAIN_TEST_SPLIT = 'train-test-split'
    EVALUATE = 'evaluate'


class EntityEmbeddingMethod(Enum):
    WORD2VEC = 'word2vec'
    STARSPACE = 'starspace'
    FASTTEXT = 'fasttext'
    PRE_TRAINED_FROM_FILE = 'pre-trained-from-file'
    DOC2VEC = 'doc2vec'


class InternalClassifier(Enum):
    LOGISTIC_REGRESSION = 'logistic-regression'
    RANDOM_FOREST = 'random-forest'
    SVC = 'svc'
    DUMMY = 'dummy'


LABEL_WORD_PREFIX = '__label__'
