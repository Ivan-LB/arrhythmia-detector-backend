"""Handling for uploaded MIT-BIH-format ECG records.

File-upload handling is a security-sensitive surface (path traversal,
resource exhaustion, malformed input). Every check here exists because an
uploaded filename and its content are both attacker-controlled and must be
validated before touching the filesystem or a parser.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import wfdb

from ecg_pipeline import preprocessing

MAX_UPLOAD_SIZE_BYTES = 200 * 1024 * 1024  # 200MB -- generous for a MIT-BIH-scale record


class InvalidRecordUpload(ValueError):
    """Raised for any upload that fails validation -- callers (the API
    layer) should translate this into a 400 response, not a 500."""


@dataclass(frozen=True)
class StoredRecord:
    record_id: str
    record_path: Path  # path to pass to wfdb.rdrecord(), without extension
    has_annotations: bool


def _safe_stem(filename: str) -> str:
    """Extract a safe base name from a client-supplied filename.

    Explicitly rejects anything with directory components or an absolute
    path ("../../../etc/230.hea", "/etc/230.hea") rather than silently
    normalizing to just "230.hea" -- suspicious input should fail loudly,
    not be quietly "fixed" into something that happens to work.

    Also rejects control characters (including NUL): Path() doesn't reject
    an embedded NUL byte at construction time, so a filename like
    "230.hea\\x00.txt" would otherwise pass every check here and only fail
    later at the OS syscall boundary in write_bytes() with an unhandled
    ValueError -- a real crash reachable from a hand-crafted multipart
    request (confirmed: httpx/browsers won't produce this, but curl and
    raw sockets will), not caught by this module's own except-and-wrap
    pattern since it happens before that try block runs.
    """
    if any(ord(character) < 0x20 for character in filename):
        raise InvalidRecordUpload(f"Invalid filename (contains control characters): {filename!r}")
    path = Path(filename)
    if path.is_absolute() or path.name != filename or not path.name or path.name in (".", ".."):
        raise InvalidRecordUpload(f"Invalid filename (must be a plain filename, no path components): {filename!r}")
    return path.stem


def _check_size(label: str, content: bytes) -> None:
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise InvalidRecordUpload(
            f"{label} exceeds the maximum upload size of {MAX_UPLOAD_SIZE_BYTES} bytes"
        )


def save_uploaded_record(
    records_root: Path,
    hea_filename: str,
    hea_content: bytes,
    dat_filename: str,
    dat_content: bytes,
    atr_filename: str | None = None,
    atr_content: bytes | None = None,
) -> StoredRecord:
    """Validate and persist an uploaded .hea/.dat pair (and optional .atr
    ground-truth annotations) to its own isolated directory, and confirm
    it's actually a readable wfdb record before returning.

    Raises InvalidRecordUpload for any validation failure: mismatched base
    names (wfdb requires them to match -- the .hea file's internal content
    references the .dat filename by that shared base name), oversized
    content, or content that fails to parse as a real record.
    """
    _check_size("hea file", hea_content)
    _check_size("dat file", dat_content)

    hea_stem = _safe_stem(hea_filename)
    dat_stem = _safe_stem(dat_filename)
    if hea_stem != dat_stem:
        raise InvalidRecordUpload(
            f"hea and dat files must share the same base name; got {hea_stem!r} and {dat_stem!r}"
        )

    has_annotations = atr_filename is not None or atr_content is not None
    if has_annotations:
        if atr_filename is None or atr_content is None:
            raise InvalidRecordUpload("atr_filename and atr_content must both be provided, or neither")
        _check_size("atr file", atr_content)
        atr_stem = _safe_stem(atr_filename)
        if atr_stem != hea_stem:
            raise InvalidRecordUpload(
                f"atr file must share the hea/dat base name; got {atr_stem!r}, expected {hea_stem!r}"
            )

    record_id = uuid.uuid4().hex
    record_dir = records_root / record_id
    record_dir.mkdir(parents=True, exist_ok=False)

    (record_dir / f"{hea_stem}.hea").write_bytes(hea_content)
    (record_dir / f"{hea_stem}.dat").write_bytes(dat_content)
    if has_annotations:
        assert atr_content is not None  # narrowed above; re-asserted for mypy across the branch
        (record_dir / f"{hea_stem}.atr").write_bytes(atr_content)

    record_path = record_dir / hea_stem
    try:
        record = wfdb.rdrecord(str(record_path))
        preprocessing.find_mlii_channel(record.sig_name)
        if has_annotations:
            wfdb.rdann(str(record_path), "atr")
    except Exception as error:
        raise InvalidRecordUpload(f"Uploaded files are not a valid wfdb record: {error}") from error

    return StoredRecord(record_id=record_id, record_path=record_path, has_annotations=has_annotations)
