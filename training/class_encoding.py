"""AAMI class <-> integer index encoding, shared by train.py and evaluate.py.

Consolidated here after a code review caught the two files independently
reimplementing this mapping, one of them (train.py's prepare_training_data)
without the validation guard the other (encode_labels) already had -- a bare
pandas .map() turns an unrecognized class into NaN, which a downstream
int/float cast can silently resolve to index 0 ("N") instead of raising.
Every caller now goes through the same validated path.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from tensorflow import keras

from ecg_pipeline.labels import AAMI_CLASSES

CLASS_TO_INDEX: dict[str, int] = {cls: idx for idx, cls in enumerate(AAMI_CLASSES)}


def class_series_to_indices(class_series: pd.Series) -> npt.NDArray[np.intp]:
    """Map AAMI class strings to integer indices (AAMI_CLASSES order).

    Raises ValueError for any value not in AAMI_CLASSES.
    """
    indices = class_series.map(CLASS_TO_INDEX)
    if indices.isna().any():
        unknown = sorted(class_series[indices.isna()].unique())
        raise ValueError(f"Unknown AAMI class(es) in data: {unknown}")
    return indices.to_numpy(dtype=np.intp)


def encode_labels(class_series: pd.Series) -> np.ndarray:
    """One-hot encode AAMI class labels in a fixed column order (AAMI_CLASSES),
    not whatever order happens to appear in a given subset -- so train and
    eval always agree on which column is which class.
    """
    indices = class_series_to_indices(class_series)
    return keras.utils.to_categorical(indices, num_classes=len(AAMI_CLASSES))
