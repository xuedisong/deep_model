from typing import List

from utils import json_util


# 特征类
class Feature(object):

    def __init__(self):
        self._index = None  # 编号
        self._name = None  # 名称
        self._type = None  # 类型
        self._valueList = []  # 值列表

    def get_index(self) -> int:
        return self._index

    def set_index(self, value: int) -> None:
        self._index = value

    def get_name(self) -> str:
        return self._name

    def set_name(self, value: str) -> None:
        self._name = value

    def get_type(self) -> str:
        return self._type

    def set_type(self, value: str) -> None:
        self._type = value

    def get_valueList(self) -> List[str]:
        return self._valueList

    def set_valueList(self, value: List[str]) -> None:
        self._valueList = value

    def __str__(self):
        return json_util.of(self)
