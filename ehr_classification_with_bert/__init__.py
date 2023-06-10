import logging
from enum import Enum

import torch

# module logger
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# available tasks
class Tasks(Enum):
    FINE_TUNE = 'fine-tune'
    EVALUATE = 'evaluate'


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


device = get_device()
