"""DS1/DS2 inter-patient train/test split for the MIT-BIH Arrhythmia Database.

Source: de Chazal, O'Dwyer & Reilly, "Automatic Classification of Heartbeats
Using ECG Morphology and Heartbeat Interval Features," IEEE Trans. Biomed.
Eng., 2004 -- the standard inter-patient evaluation protocol for MIT-BIH beat
classification. Confirmed against 3 independent sources (see
docs/data-pipeline-architecture.md).
"""

from __future__ import annotations

DS1: tuple[int, ...] = (
    101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
    201, 203, 205, 207, 208, 209, 215, 220, 223, 230,
)

DS2: tuple[int, ...] = (
    100, 103, 105, 111, 113, 117, 121, 123,
    200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234,
)

PACED_RECORDS: tuple[int, ...] = (102, 104, 107, 217)


def split_for_record(record_id: int) -> str:
    """Return "DS1" or "DS2" for a given MIT-BIH record number.

    Raises ValueError for paced records (excluded from the protocol entirely)
    or any record number outside the standard 48-record MIT-BIH set.
    """
    if record_id in DS1:
        return "DS1"
    if record_id in DS2:
        return "DS2"
    if record_id in PACED_RECORDS:
        raise ValueError(
            f"Record {record_id} contains paced beats and is excluded from the DS1/DS2 protocol."
        )
    raise ValueError(f"Record {record_id} is not part of the MIT-BIH DS1/DS2 inter-patient split.")
