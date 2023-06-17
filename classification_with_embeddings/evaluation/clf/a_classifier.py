from abc import ABC, abstractmethod
from typing import List

import numpy as np


class AClassifier(ABC):
    """Abstract base class for classifiers that defines the interface for the custom classifiers used
    for evaluation.
    """

    @abstractmethod
    def predict(self, sentences: List[List[str]]) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, sentences: List[List[str]]) -> np.ndarray:
        pass

    @abstractmethod
    def supports_predict_proba(self) -> bool:
        pass

    @abstractmethod
    def classes(self) -> list:
        pass
