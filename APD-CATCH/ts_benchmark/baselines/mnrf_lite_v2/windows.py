"""Continuous-segment aware window indexing without rewriting source rows."""

import numpy as np


def continuous_segments(index, length, expected_interval=None, tolerance=0.05, boundary_ids=None):
    """Return half-open row segments. Row-only indexes use the declared row-order protocol."""
    length = int(length)
    boundary_rows = []
    if index is not None and hasattr(index, "__len__") and len(index) != length:
        raise ValueError("timestamp length does not match row count")
    if boundary_ids is not None:
        ids = np.asarray(boundary_ids)
        if len(ids) != length:
            raise ValueError("entity/session boundary ids do not match row count")
        boundary_rows = (np.flatnonzero(ids[1:] != ids[:-1]) + 1).astype(int).tolist()
    boundaries = [0] + boundary_rows + [length]
    pieces = []
    all_gaps = []
    intervals = []
    basis = "row_order"
    for outer_start, outer_end in zip(boundaries[:-1], boundaries[1:]):
        if outer_start >= outer_end:
            continue
        if index is None:
            pieces.append((outer_start, outer_end))
            continue
        try:
            values = np.asarray(index)[outer_start:outer_end]
            if values.dtype.kind not in "mM":
                pieces.append((outer_start, outer_end))
                continue
            times = values.astype("datetime64[ns]").astype(np.int64)
        except (TypeError, ValueError, OverflowError):
            pieces.append((outer_start, outer_end))
            continue
        if np.any(times == np.iinfo(np.int64).min):
            raise ValueError("timestamps contain NaT values")
        basis = "timestamp" if not boundary_rows else "timestamp+entity_boundary"
        local_length = outer_end - outer_start
        if local_length < 2:
            pieces.append((outer_start, outer_end))
            continue
        deltas = np.diff(times)
        if np.any(deltas <= 0):
            bad = int(np.flatnonzero(deltas <= 0)[0] + 1 + outer_start)
            raise ValueError(f"timestamps duplicate or reverse at row {bad}")
        base = int(expected_interval) if expected_interval is not None else int(np.median(deltas))
        if base <= 0:
            raise ValueError("could not infer a positive normal sampling interval")
        ratios = deltas.astype(np.float64) / base
        multiples = np.maximum(1, np.rint(ratios))
        relative_error = np.abs(ratios - multiples) / multiples
        if np.any(relative_error > tolerance):
            bad = int(np.flatnonzero(relative_error > tolerance)[0] + 1 + outer_start)
            raise ValueError(
                f"unexplained timestamp interval at row {bad}: {deltas[bad - outer_start - 1]} ns, "
                f"normal interval {base} ns (tolerance {tolerance:.1%})"
            )
        gap_positions = np.flatnonzero(multiples > 1)
        starts = [outer_start] + [int(position + 1 + outer_start) for position in gap_positions]
        ends = [int(position + 1 + outer_start) for position in gap_positions] + [outer_end]
        pieces.extend(zip(starts, ends))
        intervals.append(base)
        all_gaps.extend({"left_row": int(pos + outer_start), "right_row": int(pos + outer_start + 1),
                         "interval_ns": int(deltas[pos])} for pos in gap_positions)
    if not pieces:
        pieces = [(0, 0)]
    return pieces, {
        "basis": basis,
        "expected_interval": int(np.median(intervals)) if intervals else None,
        "gaps": all_gaps,
        "entity_boundary_rows": boundary_rows,
    }


def intersect_segments(segments, start, end):
    return [(max(int(start), left), min(int(end), right))
            for left, right in segments if max(int(start), left) < min(int(end), right)]


def window_endpoints(segments, window):
    endpoints = [np.arange(left + window - 1, right, dtype=np.int64) for left, right in segments
                 if right - left >= window]
    if not endpoints:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(endpoints)
