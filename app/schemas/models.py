from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


#
# __all__ = [
#     'RequestModel',
#     'ResponseModel'
# ]
#


class ResponseModel(BaseModel):
    is_authentic: Optional[bool] = None
    score: Optional[float] = None
    original_length: Optional[int] = None
    padded_length: Optional[int] = None
    threshold: Optional[float] = None
