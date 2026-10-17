"""Continuous-segment window indexing and canonical time identities."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeIndex:
    """Canonical time coordinates; aware values are represented as UTC ns."""

    values_ns: np.ndarray
    kind: str
    source_timezone: Optional[str] = None


def normalize_time_index(index, *, explicit_time=False):
    if isinstance(index, TimeIndex):
        return index
    if isinstance(index, pd.DatetimeIndex):
        source_tz = str(index.tz) if index.tz is not None else None
        normalized = index.tz_convert("UTC") if index.tz is not None else index
        values = normalized.asi8.astype(np.int64, copy=True)
        if np.any(values == np.iinfo(np.int64).min):
            raise ValueError("timestamps contain NaT values")
        return TimeIndex(values, "timestamp_utc_ns" if source_tz else "timestamp_naive_ns", source_tz)
    if index is None:
        return TimeIndex(np.arange(0, dtype=np.int64), "row_order")
    values = np.asarray(index)
    if values.dtype.kind == "M":
        parsed = pd.DatetimeIndex(values)
        return normalize_time_index(parsed)
    if values.dtype.kind == "m":
        raise ValueError("timedelta index cannot be used as timestamps")
    if explicit_time:
        try:
            timestamps = [pd.Timestamp(value) for value in values]
            aware = [timestamp.tzinfo is not None for timestamp in timestamps]
            if any(aware) and not all(aware):
                raise ValueError("time column mixes timezone-aware and naive timestamps")
            if all(aware) and timestamps:
                parsed = pd.DatetimeIndex(pd.to_datetime(values, errors="raise", utc=True))
            else:
                parsed = pd.DatetimeIndex(timestamps)
        except Exception as error:
            raise ValueError(f"explicit time column cannot be parsed: {error}") from error
        return normalize_time_index(parsed)
    if values.dtype.kind == "O" and len(values):
        first = values[0]
        if isinstance(first, (pd.Timestamp, np.datetime64)):
            try:
                timestamps = [pd.Timestamp(value) for value in values]
                aware = [timestamp.tzinfo is not None for timestamp in timestamps]
                if any(aware) and not all(aware):
                    raise ValueError("time index mixes timezone-aware and naive timestamps")
                parsed = (pd.DatetimeIndex(pd.to_datetime(values, utc=True)) if all(aware) and timestamps
                          else pd.DatetimeIndex(timestamps))
            except Exception as error:
                raise ValueError(f"timestamp index cannot be normalized: {error}") from error
            return normalize_time_index(parsed)
    if values.dtype.kind in "iuf" or len(values) == 0:
        return TimeIndex(values.copy(), "row_order")
    raise ValueError(f"unsupported index type {values.dtype}; refusing silent row-order fallback")


def infer_expected_interval(index, length=None, boundary_ids=None, tolerance=0.05):
    normalized = normalize_time_index(index)
    if normalized.kind == "row_order":
        return None
    times = normalized.values_ns
    if length is not None and len(times) != int(length):
        raise ValueError("timestamp length does not match row count")
    if boundary_ids is None:
        groups = [(0, len(times))]
    else:
        ids = np.asarray(boundary_ids)
        if len(ids) != len(times):
            raise ValueError("entity/session boundary ids do not match row count")
        cuts = (np.flatnonzero(ids[1:] != ids[:-1]) + 1).astype(int).tolist()
        bounds = [0] + cuts + [len(times)]
        groups = list(zip(bounds[:-1], bounds[1:]))
    deltas = []
    for start, stop in groups:
        if stop - start < 2:
            continue
        local = np.diff(times[start:stop])
        if np.any(local <= 0):
            bad = int(np.flatnonzero(local <= 0)[0] + 1 + start)
            raise ValueError(f"timestamps duplicate or reverse at row {bad}")
        deltas.extend(local.tolist())
    if not deltas:
        raise ValueError("cannot infer sampling interval from fewer than two training timestamps")
    base = int(np.median(np.asarray(deltas, dtype=np.int64)))
    if base <= 0:
        raise ValueError("could not infer a positive normal sampling interval")
    # Validate that training intervals are explainable as the base cadence or gaps.
    _validate_deltas(np.asarray(deltas, dtype=np.int64), base, tolerance, 0)
    return base


def continuous_segments(index, length, expected_interval=None, tolerance=0.05, boundary_ids=None):
    """Return half-open segments; timestamp rules are applied, never re-fit, if given."""
    length = int(length)
    normalized = normalize_time_index(index) if index is not None else TimeIndex(np.arange(length), "row_order")
    if len(normalized.values_ns) != length:
        raise ValueError("time index length does not match row count")
    if boundary_ids is not None:
        ids = np.asarray(boundary_ids)
        if len(ids) != length:
            raise ValueError("entity/session boundary ids do not match row count")
        boundary_rows = (np.flatnonzero(ids[1:] != ids[:-1]) + 1).astype(int).tolist()
    else:
        boundary_rows = []
    bounds = [0] + boundary_rows + [length]
    pieces, gaps, intervals = [], [], []
    for outer_start, outer_end in zip(bounds[:-1], bounds[1:]):
        if outer_start >= outer_end:
            continue
        if normalized.kind == "row_order" or outer_end - outer_start < 2:
            pieces.append((outer_start, outer_end))
            continue
        times = normalized.values_ns[outer_start:outer_end]
        if np.any(times == np.iinfo(np.int64).min):
            raise ValueError("timestamps contain NaT values")
        deltas = np.diff(times)
        if np.any(deltas <= 0):
            bad = int(np.flatnonzero(deltas <= 0)[0] + 1 + outer_start)
            raise ValueError(f"timestamps duplicate or reverse at row {bad}")
        base = int(expected_interval) if expected_interval is not None else int(np.median(deltas))
        if base <= 0:
            raise ValueError("expected sampling interval must be positive")
        multiples = _validate_deltas(deltas, base, tolerance, outer_start)
        gap_positions = np.flatnonzero(multiples > 1)
        starts = [outer_start] + [int(position + 1 + outer_start) for position in gap_positions]
        ends = [int(position + 1 + outer_start) for position in gap_positions] + [outer_end]
        pieces.extend(zip(starts, ends))
        intervals.append(base)
        gaps.extend({"left_row": int(pos + outer_start), "right_row": int(pos + outer_start + 1),
                     "interval_ns": int(deltas[pos])} for pos in gap_positions)
    if not pieces:
        pieces = [(0, 0)]
    return pieces, {
        "basis": normalized.kind if not boundary_rows else f"{normalized.kind}+entity_boundary",
        "expected_interval": int(expected_interval) if expected_interval is not None else (int(np.median(intervals)) if intervals else None),
        "tolerance": float(tolerance),
        "source_timezone": normalized.source_timezone,
        "gaps": gaps,
        "entity_boundary_rows": boundary_rows,
    }


def _validate_deltas(deltas, base, tolerance, offset):
    ratios = deltas.astype(np.float64) / base
    multiples = np.maximum(1, np.rint(ratios))
    relative_error = np.abs(ratios - multiples) / multiples
    if np.any(relative_error > tolerance):
        bad = int(np.flatnonzero(relative_error > tolerance)[0] + 1 + offset)
        raise ValueError(
            f"unexplained timestamp interval at row {bad}: {deltas[bad - offset - 1]} ns, "
            f"normal interval {base} ns (tolerance {tolerance:.1%})"
        )
    return multiples.astype(np.int64)


def intersect_segments(segments, start, end):
    return [(max(int(start), left), min(int(end), right))
            for left, right in segments if max(int(start), left) < min(int(end), right)]


def window_endpoints(segments, window):
    endpoints = [np.arange(left + window - 1, right, dtype=np.int64) for left, right in segments
                 if right - left >= window]
    if not endpoints:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(endpoints)
