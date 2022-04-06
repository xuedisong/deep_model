from typing import TypeVar, List

T = TypeVar('T')


def ofList(it) -> 'List[T]':
    if it is None:
        return []
    return it
