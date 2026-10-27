"""
Custom text parser for the blood-pressure datasets.

Loaded by petsard/processor/text_parser.py, which is wired into the pipeline
from YAML::

    parser_path: petsard/synthesizer/LLM_synth/utils/bloodpressure_text_parser.py

The only contract is the entry function::

    def text_parser(df: pd.DataFrame) -> pd.DataFrame

It receives a whole frame (original, synthetic or control -- the same function
runs on all of them) and returns a frame with one row per input row. Anything
inside is yours to change; edit the FIELD regexes below to fit other datasets,
or copy this file as a template for a different domain.

Handles both bundled layouts:
    real_bloodpressure.csv   患者調查：名稱=陳柏宇，性別=男，年齡=45，血壓=138/88 mmHg
    real_bloodpressure2.csv  患者為45歲男性，姓名：陳柏宇，血壓138/88偏高

Caveat on real_bloodpressure.csv: it carries no 偏高/正常 wording, so bp_label
is genuinely absent there and will report 0.0 coverage. That is the parse
coverage report doing its job -- drop the field from this parser, or use
real_bloodpressure2.csv, rather than reading the zero as a synthesis failure.
"""

from __future__ import annotations

import re

import pandas as pd

# Column holding the natural-language text. Falls back to the first object
# column when this name is absent, so a renamed column still parses.
SOURCE_COLUMN = "text"

# Keep the original text alongside the parsed fields. Turn off for a purely
# tabular output -- note that evaluators score every column they are given, and
# free text scores badly for reasons that are not quality problems.
KEEP_SOURCE = False

# --- extraction patterns -----------------------------------------------------

# Age, tried in order. Both forms anchor on a keyword so they cannot swallow
# blood-pressure digits: "45歲" in the prose layout, "年齡=45" in the field layout.
AGE_PATTERNS = [
    re.compile(r"(\d{1,3})\s*歲"),
    re.compile(r"年齡[=：:]\s*(\d{1,3})"),
]

# Systolic / diastolic from the NNN/NN form.
BP_RE = re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})")

# Gender. Ordered: the first label with any matching pattern wins. Patterns are
# specific on purpose -- a bare 男/女 search would misread "男性患者的女兒".
GENDER_PATTERNS: list[tuple[str, list[re.Pattern]]] = [
    (
        "女",
        [
            re.compile(r"性別[=：:]?\s*女"),
            re.compile(r"女性"),
            re.compile(r"[（(]\s*女"),
        ],
    ),
    (
        "男",
        [
            re.compile(r"性別[=：:]?\s*男"),
            re.compile(r"男性"),
            re.compile(r"[（(]\s*男"),
        ],
    ),
]

# Blood-pressure reading label.
LABEL_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("偏高", re.compile(r"偏高")),
    ("正常", re.compile(r"正常")),
]

# Patient name. Not extracted by default -- see NOTE at the bottom of the file.
NAME_RE = re.compile(r"(?:姓名|名稱)[=：:]\s*([一-鿿]{2,4})")
EXTRACT_NAME = False


def _first_label(text: str, table) -> str | None:
    """Return the first label whose pattern(s) match, else None."""
    for label, patterns in table:
        if isinstance(patterns, re.Pattern):
            patterns = [patterns]
        if any(p.search(text) for p in patterns):
            return label
    return None


def _parse_row(text: object) -> dict:
    """Extract every field from one cell. Missing fields come back as None."""
    empty = {
        "gender": None,
        "age": None,
        "systolic": None,
        "diastolic": None,
        "bp_label": None,
    }
    if EXTRACT_NAME:
        empty["name"] = None

    if text is None or (isinstance(text, float) and pd.isna(text)):
        return empty
    source = str(text).strip()
    if not source:
        return empty

    record = dict(empty)
    record["gender"] = _first_label(source, GENDER_PATTERNS)
    record["bp_label"] = _first_label(source, LABEL_PATTERNS)

    for pattern in AGE_PATTERNS:
        age = pattern.search(source)
        if age:
            record["age"] = int(age.group(1))
            break

    bp = BP_RE.search(source)
    if bp:
        record["systolic"] = int(bp.group(1))
        record["diastolic"] = int(bp.group(2))

    if EXTRACT_NAME:
        name = NAME_RE.search(source)
        if name:
            record["name"] = name.group(1)

    return record


def _resolve_source_column(df: pd.DataFrame) -> str:
    """Pick the text column, tolerating a rename."""
    if SOURCE_COLUMN in df.columns:
        return SOURCE_COLUMN
    object_columns = [c for c in df.columns if df[c].dtype == "object"]
    if not object_columns:
        raise KeyError(
            f"No column '{SOURCE_COLUMN}' and no text column to fall back on. "
            f"Columns present: {list(df.columns)}"
        )
    return object_columns[0]


def text_parser(df: pd.DataFrame) -> pd.DataFrame:
    """Parse a frame of natural-language records into typed columns.

    Args:
        df: One frame from the pipeline (ori, syn or control), holding the
            free-text column named by SOURCE_COLUMN.

    Returns:
        A frame with the same index and one column per extracted field. Numeric
        columns are int64 when every row parsed, float64 when any row missed
        (pandas cannot hold NaN in int64) -- the caller unifies dtypes across
        frames afterwards, so a miss in one frame does not break comparison.
    """
    column = _resolve_source_column(df)

    records = [_parse_row(value) for value in df[column]]
    parsed = pd.DataFrame(records, index=df.index)

    for numeric in ("age", "systolic", "diastolic"):
        values = pd.to_numeric(parsed[numeric], errors="coerce")
        parsed[numeric] = (
            values.astype("int64") if values.notna().all() else values.astype("float64")
        )

    if KEEP_SOURCE:
        parsed.insert(0, column, df[column])

    return parsed


# NOTE on the name field
# ---------------------
# EXTRACT_NAME is off by default. A patient name is a free identifier: every
# synthetic value is novel, so SDMetrics' CategoryAdherence scores it 0. That
# zero is the *desired* privacy behaviour (no verbatim copying of training
# records) but it reads as a validity failure and drags the Data Validity
# average down. Turn it on when running Anonymeter, where a quasi-identifier is
# exactly what you want to attack with.
