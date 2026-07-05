import numpy as np
import pandas as pd
import pytest

from ecg_pipeline.labels import AAMI_CLASSES
from training.class_encoding import class_series_to_indices, encode_labels


class TestClassSeriesToIndices:
    def test_maps_known_classes_to_the_aami_classes_order(self):
        indices = class_series_to_indices(pd.Series(["N", "V", "S"]))
        expected = [AAMI_CLASSES.index("N"), AAMI_CLASSES.index("V"), AAMI_CLASSES.index("S")]
        np.testing.assert_array_equal(indices, expected)

    def test_raises_clearly_for_an_unknown_class_instead_of_silently_mapping_to_zero(self):
        # A bare pandas .map() would turn an unrecognized label into NaN,
        # which a downstream int/float cast can silently resolve to index 0
        # ("N") -- this must raise instead, everywhere this mapping happens.
        with pytest.raises(ValueError, match="Unknown AAMI class"):
            class_series_to_indices(pd.Series(["N", "TYPO"]))

    def test_returns_integer_dtype_not_float(self):
        # A NaN-containing intermediate promotes the array to float64; even
        # on the valid-data path, the result must be a clean integer dtype.
        indices = class_series_to_indices(pd.Series(["N", "V"]))
        assert np.issubdtype(indices.dtype, np.integer)


class TestEncodeLabels:
    def test_one_hot_shape_matches_number_of_aami_classes(self):
        encoded = encode_labels(pd.Series(["N", "V", "S"]))
        assert encoded.shape == (3, len(AAMI_CLASSES))

    def test_column_order_matches_aami_classes_order(self):
        encoded = encode_labels(pd.Series(["N"]))
        expected = [1.0 if cls == "N" else 0.0 for cls in AAMI_CLASSES]
        np.testing.assert_array_equal(encoded[0], expected)

    def test_raises_clearly_for_an_unknown_class(self):
        with pytest.raises(ValueError, match="Unknown AAMI class"):
            encode_labels(pd.Series(["N", "X"]))
