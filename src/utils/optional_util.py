from functools import reduce
from typing import TypeVar, Callable, List, Set, Generic, Dict, Iterable, Optional, Any
from itertools import islice, chain

T = TypeVar('T')
R = TypeVar('R')
K = TypeVar('K')
U = TypeVar('U')


class Option(Generic[T]):

    def __init__(self, value: T):
        self._value: T = value

    @staticmethod
    def of(value: T) -> 'Option[T]':
        assert value is not None
        return Option(value)

    @staticmethod
    def ofNullable(value: T) -> 'Option[T]':
        return Option(value)

    def ifPresent(self, func: Callable[[T], None]) -> None:
        if self._value is not None:
            func(self._value)

    def map(self, func: Callable[[T], R]) -> 'Option[R]':
        if self._value is None:
            return Option.ofNullable(None)
        return Option.ofNullable(func(self._value))

    def __str__(self):
        return self._value


EMPTY = Option.ofNullable(None)
