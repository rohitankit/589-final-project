from typing import List, Dict, Tuple, Any, Union, Callable, TypeVar, Generic, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


@dataclass
class DictDataset:
    attributes: Dict[str, Any]
    labels: Dict[int, Any]


@dataclass
class VectorDataset:
    """This class stores the attributes and labels for a dataset where the attributes can be
    represented by a vector, and the labels are a probability distribution over the classes
    """

    attributes: np.ndarray
    labels: np.ndarray

T = TypeVar("T")


class ValueVectorizer(ABC, Generic[T]):
    vector_size: int

    @abstractmethod
    def __init__(self, values: List[T], **kwargs):
        pass

    @abstractmethod
    def vectorizer_fn(self, data: T) -> np.ndarray:
        pass

    def __call__(self, arg: T) -> np.ndarray:
        return self.vectorizer_fn(arg)


class StrVectorizer(ValueVectorizer[str]):
    def __init__(self, values: List[str]):
        self.values = list(set(values))

        self.vector_size = len(self.values)

    def vectorizer_fn(self, arg: str) -> np.ndarray:
        result = np.zeros(self.vector_size, dtype=np.float64)

        if arg in self.values:
            result[self.values.index(arg)] = 1.0

        return result


class FloatVectorizer(ValueVectorizer[float]):
    def __init__(self, values: List[float], normalize: bool = True):
        self.values = list(set(values))
        self.categorical = len(self.values) < 5
        self.normalize = normalize

        if self.categorical:
            self.vector_size = len(self.values)
        else:
            if normalize:
                self.data_min = min(self.values)
                self.data_range = max(self.values) - self.data_min

            self.values = None

            self.vector_size = 1

    def vectorizer_fn(self, arg: float) -> np.ndarray:
        result = np.zeros(self.vector_size, dtype=np.float64)

        if self.categorical:
            if self.values is not None and arg in self.values:
                result[self.values.index(arg)] = 1.0

            return result

        else:
            if self.normalize:
                result[0] = (arg - self.data_min) / self.data_range
            else:
                result[0] = arg

            return result


class IntVectorizer(ValueVectorizer[int]):
    def __init__(self, values: List[int], normalize: bool = True):
        self.values = list(set(values))
        self.categorical = len(self.values) < 5
        self.normalize = normalize

        if self.categorical:
            self.vector_size = len(self.values)
        else:
            if normalize:
                self.data_min = min(self.values)
                self.data_range = max(self.values) - self.data_min

            self.values = None

            self.vector_size = 1

    def vectorizer_fn(self, arg: int) -> np.ndarray:
        result = np.zeros(self.vector_size, dtype=np.float64)

        if self.categorical:
            if self.values is not None and arg in self.values:
                result[self.values.index(arg)] = 1.0

            return result

        else:
            if self.normalize:
                result[0] = (arg - self.data_min) / self.data_range
            else:
                result[0] = arg

            return result


class BoolVectorizer(ValueVectorizer[bool]):
    def __init__(self, values: List[str]):
        self.vector_size = 1

    def __call__(self, arg: bool) -> np.ndarray:
        result = np.zeros(self.vector_size, dtype=np.float64)

        result[0] = float(arg)

        return result


data_vectorizer_types = {
    int: IntVectorizer,
    float: FloatVectorizer,
    str: StrVectorizer,
    bool: BoolVectorizer,
}


def get_vectorizer(dtype: Type) -> Type[ValueVectorizer]:
    vectorizer = data_vectorizer_types.get(dtype)

    if vectorizer is None:
        raise ValueError(f"No vectorizer implemented for dtype {dtype}.")

    return vectorizer


class DataVectorizer:
    def __init__(self, data: List[List], label_index: int = -1):
        self.label_index = label_index
        self.vectorizers: List[ValueVectorizer] = []

        num_columns = len(data[0])

        for attribute_idx in range(num_columns):
            attribute_values = [row[attribute_idx] for row in data]
            attribute_dtype = type(attribute_values[0])
            data_vectorizer_type = get_vectorizer(attribute_dtype)

            vectorizer = data_vectorizer_type(attribute_values)
            self.vectorizers.append(vectorizer)

    def vectorize_dataset(self, data: List[List]) -> VectorDataset:
        """This function vectorizes an entire dataset using the vectorizers made during
        initialization

        Args:
            data (DictDataset): A dataset to vectorize

        Returns:
            Tuple[np.ndarray, np.ndarray]: (x_result, y_result) the vectors for the attributes and
            labels
        """

        y_vector_length = self.vectorizers[self.label_index].vector_size
        x_vector_length = (
            sum(vectorizer.vector_size for vectorizer in self.vectorizers)
            - y_vector_length
        )

        num_examples = len(data)

        x_result = np.zeros((num_examples, x_vector_length), dtype=np.float64)
        y_result = np.zeros((num_examples, y_vector_length), dtype=np.float64)

        for example_idx, example in enumerate(data):
            
            example_x_vector, example_y_vector = self.vectorize_datapoint(example)

            x_result[example_idx] = example_x_vector
            y_result[example_idx] = example_y_vector

        return VectorDataset(attributes=x_result, labels=y_result)

    def vectorize_datapoint(self, example: List):
        attribute_vectors = [
            vectorizer(attribute)
            for vectorizer, attribute in zip(self.vectorizers, example)
        ]

        label_vector = attribute_vectors[self.label_index]
        del attribute_vectors[self.label_index]

        full_attribute_vector = np.concatenate(attribute_vectors)

        return full_attribute_vector, label_vector
