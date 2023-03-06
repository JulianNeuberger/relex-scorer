import typing

import model

from augment.base import BaseAugmenter
from augment.example import ExampleAugmenter

augmenters: typing.Dict[str, typing.Type[BaseAugmenter]] = {
    'example': ExampleAugmenter
}


def augment_dataset(dataset: model.DataSet, augmentation_plan: typing.List[BaseAugmenter]) -> model.DataSet:
    for augmenter in augmentation_plan:
        dataset = augmenter.augment(dataset)
    return dataset
