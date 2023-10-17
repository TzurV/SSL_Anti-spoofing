import random

from app.core.config import local_settings
from app.schemas import ResponseModel


def demo(utterance) -> ResponseModel:
    response = ResponseModel()
    response.original_length = utterance.shape[0]
    response.score = random.uniform(-1, 1)
    response.threshold = local_settings["threshold"]
    if response.score > response.threshold:
        response.is_authentic = True
    else:
        response.is_authentic = False
    return response
