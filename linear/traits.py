import abc
from abc import ABC, abstractmethod

class FeatureSelectableTrait(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def squash(self):
        pass
    @abstractmethod
    def alpha_feature_selectors(self):
        pass
    @abstractmethod
    def feature_normalizers(self):
        pass