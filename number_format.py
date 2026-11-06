"""Thousands-separator formatting for numbers shown in dashboard tables.

The numeric fields in the NDC data are mostly free-text strings
("2,555 in 2030; 5,110 in 2035", "75,000", "-2889"), so a NumberColumn
format alone cannot cover them. `thousands()` walks any scalar (number or
string) and groups the integer part of every bare number >= 1,000 with
thousands separators. Numbers that already carry separators ("28,934"),
decimals ("12.3") and 4-digit year-like tokens (1900-2099) are left
untouched, so "2030" never becomes "2,030".
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# A bare integer part of 4+ digits: not preceded by a digit/comma/dot (so
# the groups of an already-formatted number or the fraction of a decimal
# are never matched), optionally followed by a decimal part, and not
# followed by a digit or comma.
_BARE_NUM = re.compile(r"(?<![\d.,])(\d{4,})(\.\d+)?(?![\d,.])")


def _sub(m: re.Match) -> str:
    int_part, dec = m.group(1), m.group(2) or ""
    n = int(int_part)
    if not dec and 1900 <= n <= 2099:  # a year, not a quantity
        return m.group(0)
    return f"{n:,}{dec}"


def thousands(val):
    """Return `val` with thousands separators added to bare numbers >= 1000.

    None / NaN / booleans pass through unchanged; plain ints and floats are
    returned as grouped strings ("9073.37" -> "9,073.37"); strings get every
    bare number formatted in place, leaving surrounding text as-is.
    """
    if val is None or isinstance(val, bool):
        return val
    if isinstance(val, (int, float, np.integer, np.floating)):
        if pd.isna(val) or (
            isinstance(val, (float, np.floating)) and not np.isfinite(val)
        ):
            return val
        if isinstance(val, (float, np.floating)) and float(val).is_integer():
            return f"{int(val):,}"  # 227000.0 -> "227,000", not "227,000.0"
        return f"{val:,}"
    if isinstance(val, str):
        return _BARE_NUM.sub(_sub, val)
    return val


def thousands_df(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Copy of `df` with `thousands()` applied to the given `columns`.

    Default (columns=None): every non-boolean numeric column.
    """
    out = df.copy()
    if columns is None:
        columns = [
            c for c in out.columns
            if pd.api.types.is_numeric_dtype(out[c])
            and not pd.api.types.is_bool_dtype(out[c])
        ]
    for c in columns:
        if c in out.columns:
            out[c] = out[c].map(thousands)
    return out
