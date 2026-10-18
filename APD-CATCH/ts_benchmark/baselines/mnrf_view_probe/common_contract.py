"""Pure helpers shared by the Probe common-contract producer and consumers."""

import numpy as np


CONTRACT_SCHEMA = "mnrf-view-probe-common-export-v2"
REQUIRED_CONTRACT_FIELDS = (
    "probe_run", "probe_manifest_sha256", "probe_source_identity",
    "data_identity", "time_identity", "time_kind", "boundary_columns",
    "temporal_rule", "ranges", "segment_ranges", "test_segments",
    "catch_splits", "catch_test", "train_rows", "labels",
)


def missing_contract_fields(contract):
    return [field for field in REQUIRED_CONTRACT_FIELDS if field not in contract]


def time_semantics_compatible(expected_kind, expected_timezone, actual_kind, actual_timezone):
    """Compare time coordinates by meaning, not by a display timezone name.

    ``timestamp_utc_ns`` is already canonical absolute time.  A CSV round trip
    may therefore change ``Asia/Shanghai`` to ``UTC`` without changing the
    contract.  Naive timestamps and row-order coordinates have no such
    absolute-time guarantee and keep their original metadata strict.
    """
    if str(expected_kind) != str(actual_kind):
        return False
    kind = str(expected_kind)
    if kind == "timestamp_utc_ns":
        return True
    return expected_timezone == actual_timezone


def endpoints_for_ranges(segment_ranges, lower, upper, seq_len):
    """Derive the complete endpoint sequence without subsampling."""
    result = []
    for left, right in segment_ranges:
        start = max(int(left), int(lower))
        stop = min(int(right), int(upper))
        first = start + int(seq_len) - 1
        if first < stop:
            result.extend(range(first, stop))
    return np.asarray(result, dtype=np.int64)


def expected_contract_endpoints(contract, split, seq_len=None):
    source = contract["probe_source_identity"]
    length = int(contract.get("catch_seq_len") if seq_len is None else seq_len)
    if split == "test":
        return endpoints_for_ranges(source.get("test_segments", []),
                                    int(source.get("train_length", -1)),
                                    int(source.get("input_length", 0)), length)
    raw_range = source.get("ranges", {}).get(split)
    segments = source.get("segment_ranges", {}).get(split)
    if not raw_range or not segments:
        raise ValueError(f"common contract lacks frozen {split} ranges/segments")
    return endpoints_for_ranges(segments, raw_range[0], raw_range[1], length)
