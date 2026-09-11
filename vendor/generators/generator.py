from abc import ABC, abstractmethod

class Generator(ABC):
    @abstractmethod
    def generate(self, level):
        raise NotImplementedError("Subclasses must implement this method")