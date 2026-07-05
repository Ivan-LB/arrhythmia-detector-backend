import pytest

from ecg_pipeline import splits


def test_ds1_has_22_records():
    assert len(splits.DS1) == 22


def test_ds2_has_22_records():
    assert len(splits.DS2) == 22


def test_paced_records_has_4_records():
    assert len(splits.PACED_RECORDS) == 4


def test_ds1_ds2_and_paced_cover_all_48_mitbih_records_with_no_overlap():
    ds1_set = set(splits.DS1)
    ds2_set = set(splits.DS2)
    paced_set = set(splits.PACED_RECORDS)

    assert ds1_set.isdisjoint(ds2_set)
    assert ds1_set.isdisjoint(paced_set)
    assert ds2_set.isdisjoint(paced_set)
    assert len(ds1_set | ds2_set | paced_set) == 48


def test_paced_records_are_excluded_from_both_splits():
    for record in splits.PACED_RECORDS:
        assert record not in splits.DS1
        assert record not in splits.DS2


@pytest.mark.parametrize("record_id", [101, 106, 230])
def test_split_for_record_identifies_ds1_records(record_id):
    assert splits.split_for_record(record_id) == "DS1"


@pytest.mark.parametrize("record_id", [100, 111, 234])
def test_split_for_record_identifies_ds2_records(record_id):
    assert splits.split_for_record(record_id) == "DS2"


def test_split_for_record_raises_for_paced_record():
    with pytest.raises(ValueError, match="paced"):
        splits.split_for_record(102)


def test_split_for_record_raises_for_record_not_in_mitbih():
    with pytest.raises(ValueError, match="not part of"):
        splits.split_for_record(999)
