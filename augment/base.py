import abc
import typing

import model


class BaseAugmenter(abc.ABC):
    def __init__(self, **kwargs):
        pass

    @staticmethod
    def get_parameters() -> typing.Dict[str, typing.Type]:
        return {}

    def augment(self, dataset: model.DataSet) -> model.DataSet:
        raise NotImplementedError()
