import numpy as np

from BOB.inference.utils import Prompt

class BaseSegmenter:
    def __init__(self):
        pass
    
    def predict(self, image, prompts: list[Prompt]) -> tuple[list[np.ndarray], list[float], list[Prompt]]:
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_model_name(self) -> str:
        raise NotImplementedError("Subclasses should implement this method.")