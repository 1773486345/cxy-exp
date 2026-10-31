import io

import numpy as np
import pandas as pd

from scripts.build_calit2_raw_count import parse_uci_data


def test_calit2_raw_count_parser_preserves_two_digit_counts():
    times = pd.date_range("2005-07-24", periods=5040, freq="30min")
    rows = []
    for index, timestamp in enumerate(times):
        value7 = 62 if index == 0 else index % 55
        value9 = 54 if index == 1 else index % 63
        date = timestamp.strftime("%m/%d/%y")
        clock = timestamp.strftime("%H:%M:%S")
        rows.extend([[7, date, clock, value7], [9, date, clock, value9]])
    payload = pd.DataFrame(rows).to_csv(index=False, header=False).encode()
    parsed = parse_uci_data(payload)
    assert int(parsed.iloc[0][7]) == 62
    assert int(parsed.iloc[1][9]) == 54
    assert parsed.dtypes.tolist() == [np.dtype("int64"), np.dtype("int64")]
