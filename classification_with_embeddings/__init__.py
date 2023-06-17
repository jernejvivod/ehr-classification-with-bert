import logging
from enum import Enum

import torch

__version__ = "0.1.0"

# module logger

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Tasks(Enum):
    GET_ENTITY_EMBEDDINGS = 'get-entity-embeddings'
    TRAIN_CNN_MODEL = 'train-cnn-model'
    TRAIN_TEST_SPLIT = 'train-test-split'
    EVALUATE_EMBEDDINGS_MODEL = 'evaluate-embeddings-model'
    EVALUATE_CNN_MODEL = 'evaluate-cnn-model'


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

torch_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
