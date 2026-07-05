import pytest

from ecg_pipeline import labels


def test_aami_classes_are_the_five_superclasses():
    assert labels.AAMI_CLASSES == ("N", "S", "V", "F", "Q")


@pytest.mark.parametrize(
    "symbol,expected_class",
    [
        ("N", "N"),
        ("L", "N"),
        ("R", "N"),
        ("e", "N"),
        ("j", "N"),
        ("A", "S"),
        ("a", "S"),
        ("J", "S"),
        ("S", "S"),
        ("V", "V"),
        ("E", "V"),
        ("F", "F"),
        ("/", "Q"),
        ("f", "Q"),
        ("Q", "Q"),
    ],
)
def test_to_aami_class_maps_every_known_symbol(symbol, expected_class):
    assert labels.to_aami_class(symbol) == expected_class


def test_to_aami_class_returns_none_for_non_beat_annotations():
    # e.g. rhythm-change markers, not beat symbols
    assert labels.to_aami_class("+") is None
    assert labels.to_aami_class("~") is None


def test_every_mapped_symbol_resolves_to_a_valid_aami_class():
    for symbol in labels.SYMBOL_TO_AAMI:
        assert labels.to_aami_class(symbol) in labels.AAMI_CLASSES
