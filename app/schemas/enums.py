from enum import Enum

__all__ = [
    'OperationType'
]


class OperationType(str, Enum):
    Add = 'Add'
    Sub = 'Sub'
    Mult = 'Mult'
    Div = 'Div'
