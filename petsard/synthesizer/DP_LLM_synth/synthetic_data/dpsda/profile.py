# Part of the differentially private LLM synthesizer added in 2026; not adapted from Aug-PE.
# Licensed under the MIT License of this repository (see LICENSE).

"""Step 0 of the DP LLM synthesizer: a differentially private data profile.

Replaces hand-written descriptions, formats and public examples, for any kind of text:
structured reports, plain-text orders, free-form reviews. Fixed rules that assume no format
turn every private row into items of four families; each family is one Gaussian mechanism:

  structure  labels (a line start up to a colon, or a name followed by ": " inside a line)
             and runs of a repeated symbol such as ___  -- empty when the data have none
  per-row    "column=value" for every short field value (e.g. labels, ratings), the
             combination of all short values of the row, and the row's whole label sequence
             (its template); combinations and templates are drawn later as one unit
  vocabulary frequent words and two-word phrases (two-character sequences for CJK text)
  shape      rows per (length bin, line-count bin)

For the first three families every row adds 1 to each of at most `cap` items (sensitivity
sqrt(cap)); Gaussian noise is added and only items whose noisy count exceeds a threshold
are released, so an item held by a single row is released with probability <= delta_s
(a Gaussian set union). The shape histogram has sensitivity 1; its total is the noisy row
count.

A fifth mechanism (optional) links text to values: on the domain released above (short
column values x vocabulary items) every row adds 1 to (its value of each short column,
each vocabulary item it holds), at most `max_vocab` items per row; Gaussian noise goes on
every cell of that fixed domain. Post-processing keeps, per column value, the items that
are clearly more frequent among rows with that value. Only noisy outputs leave this
module: raw items and exact counts stay in memory.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

import numpy as np

from .accountant import threshold_counts
from .dp_random import dp_rng, dp_sys_random

SHORT_VALUE_MAX = 20      # a field value this short becomes a "column=value" item
LABEL_MAX = 40            # longest line label kept
SYMBOL_RUN_MAX = 10       # longer runs of one symbol are cut to this length
LENGTH_EDGES = [0] + [2 ** k for k in range(4, 17)]   # 0, 16, 32, ..., 65,536 characters
LINE_EDGES = [1, 2, 3, 5, 9, 17, 33, 65]              # 1, 2, 3-4, 5-8, ..., 65+ lines

_CJK = r"一-鿿㐀-䶿豈-﫿぀-ヿ가-힯"
MAX_TEMPLATE = 30         # labels kept in one row's label sequence
NO_LABELS = "(none)"      # the label sequence of a row without labels
WORDING_MIN_LIFT = 1.5    # a value's typical wording: at least 1.5x as frequent as overall
WORDING_MIN_SNR = 3.0     # ... with a noisy count of at least 3 noise standard deviations
WORDING_TOP = 12          # items kept per column value

# a label at the start of a line (may hold spaces, e.g. "CHIEF COMPLAINT:", but no sentence
# punctuation, so an ordinary sentence is never taken for a label) ...
_PUNCT = r",.;!?，。；！？"
_LABEL_RE = re.compile(r"^[ \t]*([^\s:：%s][^:：\n%s]{0,%d})([:：])" % (_PUNCT, _PUNCT, LABEL_MAX - 1), re.M)
# ... or a single name followed by a colon and a space inside a line ("TCP: 59952");
# "https://", "13:44" and '"type":' are not labels
_INLINE_LABEL_RE = re.compile(r"(?<![\w/.:\-])([^\W\d_][\w./\-]{0,29})([:：])(?=[ \t]|$)", re.M)
_SYMBOL_RE = re.compile(r"((?:[^\w\s]|_))\1+")
_TOKEN_RE = re.compile(rf"[{_CJK}]+|[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
_CJK_RUN = re.compile(rf"^[{_CJK}]+$")
_HAS_LETTER = re.compile(r"[^\W\d_]")

PROFILE_LABELS = {
    "label": "Labels used in the data (a name followed by a colon)",
    "symbol": "Symbol patterns used in the data",
    "word": "Frequent words",
    "phrase": "Frequent phrases",
}


def _is_missing(value) -> bool:
    try:
        return value is None or (isinstance(value, float) and math.isnan(value))
    except TypeError:
        return False


def find_labels(text: str) -> list[str]:
    """Labels in order of appearance: line-start labels and inline "name: " labels.
    A label must contain a letter of some script ("[13:" from a time stamp is none)."""
    found: dict[int, str] = {}
    line_label_ends = set()
    for m in _LABEL_RE.finditer(text):
        # an ASCII colon must be followed by a space or the line end ("12:30" is a time)
        after = text[m.end(2):m.end(2) + 1]
        if m.group(2) == ":" and after not in ("", " ", "\t", "\r", "\n"):
            continue
        if _HAS_LETTER.search(m.group(1)):
            found[m.start(1)] = m.group(1).rstrip() + m.group(2)
            line_label_ends.add(m.end(2))
    for m in _INLINE_LABEL_RE.finditer(text):
        if m.end(2) not in line_label_ends:  # not the tail of a line-start label
            found.setdefault(m.start(1), m.group(1) + m.group(2))
    return [found[k] for k in sorted(found)]


def structure_items(text: str) -> tuple[list[str], set[tuple[str, str]]]:
    labels = find_labels(text)
    items = {("label", x) for x in labels}
    items |= {("symbol", m.group(0)[:SYMBOL_RUN_MAX]) for m in _SYMBOL_RE.finditer(text)}
    return labels, items


def vocabulary_items(text: str) -> set[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    for line in text.splitlines():
        prev = None
        for tok in _TOKEN_RE.findall(line):
            if _CJK_RUN.match(tok):
                items |= {("word", tok[i:i + 2]) for i in range(len(tok) - 1)}
                prev = None
                continue
            if tok.isdigit() or len(tok) < 2:
                prev = None
                continue
            items.add(("word", tok))
            if prev is not None:
                items.add(("phrase", f"{prev} {tok}"))
            prev = tok
    return items


def row_items(row: dict, joint_fields: bool = True) -> tuple[set, set, set]:
    """(structure, per-row, vocabulary) items of one row.

    Short values become "column=value" items; long values contribute labels, symbol runs
    and vocabulary. The combination of all short values of the row (when it has two or more)
    and the row's whole label sequence (its "template") join the per-row family, which holds
    at most one item per column plus these two, so its shares are never thinned by a cap."""
    structure, per_row, vocab = set(), set(), set()
    labels: list[str] = []
    short: list[list[str]] = []
    has_text = False
    for col, value in row.items():
        text = "" if _is_missing(value) else str(value)
        if len(text) <= SHORT_VALUE_MAX:
            per_row.add(("field", f"{col}={text}"))
            short.append([str(col), text])
        else:
            has_text = True
            found, items = structure_items(text)
            labels += found
            structure |= items
            vocab |= vocabulary_items(text)
    if joint_fields and len(short) >= 2:
        per_row.add(("combo", json.dumps(short, ensure_ascii=False)))
    if has_text:
        per_row.add(("template", " → ".join(labels[:MAX_TEMPLATE]) if labels else NO_LABELS))
    return structure, per_row, vocab


def _bin(value: int, edges: list[int]) -> int:
    for i in range(len(edges) - 1, -1, -1):
        if value >= edges[i]:
            return i
    return 0


def _release(item_sets, cap, sigma, delta_s, noise, pick):
    """Thresholded noisy counts: {item: noisy count} of the released items, and the threshold."""
    counts: Counter = Counter()
    for s in item_sets:
        s = sorted(s)
        if len(s) > cap:
            s = pick.sample(s, cap)
        counts.update(s)
    tau = threshold_counts(sigma, delta_s, cap)
    keys = sorted(counts)
    if not keys:
        return {}, tau
    noisy = np.array([counts[k] for k in keys], dtype=float) + noise.normal(0.0, sigma * math.sqrt(cap), len(keys))
    return {k: float(v) for k, v in zip(keys, noisy) if v > tau}, tau


def n_queries(conditional_vocab: bool, wording_own_budget: bool = False) -> int:
    """Gaussian mechanisms sharing the profile budget: structure, per-row, vocabulary, shape,
    plus the value wording unless it has a budget of its own."""
    return 5 if conditional_vocab and not wording_own_budget else 4


def _value_wording(field_sets, vocab_sets, released_fields, released_vocab, sigma, max_vocab,
                   noise, pick):
    """Fifth mechanism: noisy counts of (column value, vocabulary item) on a domain fixed by
    the released outputs, then the items clearly more frequent among rows with each value.

    Each row adds 1 to (its value of each short column, each released item it holds), at
    most max_vocab items, so the L2 sensitivity is sqrt(#columns x max_vocab); the domain
    is public (released), so every cell gets noise and no threshold is needed."""
    values: dict[str, list[str]] = {}
    for kind, item in sorted(released_fields):
        if kind == "field":
            col, value = item.split("=", 1)
            values.setdefault(col, []).append(value)
    vocab = sorted(released_vocab)
    if not values or not vocab:
        return {}, 0.0, None
    index = {k: i for i, k in enumerate(vocab)}
    cells = {(c, v): np.zeros(len(vocab)) for c, vs in values.items() for v in vs}
    for fs, vs in zip(field_sets, vocab_sets):
        items = sorted(k for k in vs if k in index)
        if len(items) > max_vocab:
            items = pick.sample(items, max_vocab)
        if not items:
            continue
        idx = [index[k] for k in items]
        for kind, item in fs:
            if kind == "field":
                key = tuple(item.split("=", 1))
                if key in cells:
                    cells[key][idx] += 1
    std = sigma * math.sqrt(len(values) * max_vocab)
    for key in cells:
        cells[key] = cells[key] + noise.normal(0.0, std, len(vocab))
    # the noisy counts themselves are a DP output: kept for drawing words per row
    counts = {"items": [list(k) for k in vocab],
              "values": {c: {v: [round(float(x), 3) for x in cells[(c, v)]] for v in vs}
                         for c, vs in values.items()}}

    # post-processing: within each column, an item's share among rows with a value
    # against its share over all values of that column
    wording: dict[str, dict[str, list]] = {}
    for col, vs in values.items():
        mat = np.clip(np.vstack([cells[(col, v)] for v in vs]), 0.0, None)
        grand, item_tot = mat.sum(), mat.sum(axis=0)
        for r, value in enumerate(vs):
            row_tot = mat[r].sum()
            if row_tot <= 0 or grand <= 0:
                continue
            expected = row_tot * item_tot / grand
            lift = np.divide(mat[r], expected, out=np.zeros_like(expected), where=expected > 0)
            ok = np.nonzero((mat[r] >= WORDING_MIN_SNR * std) & (lift >= WORDING_MIN_LIFT))[0]
            best = sorted(ok, key=lambda i: -lift[i])[:WORDING_TOP]
            if best:
                wording.setdefault(col, {})[value] = [
                    {"item": vocab[i][1], "lift": round(float(lift[i]), 2)} for i in best]
    return wording, std, counts


def build_profile(rows: list[dict], row_texts: list[str], columns: list[str], sigma: float,
                  delta_s: float, max_items: int = 32, max_vocab: int = 64,
                  joint_fields: bool = True, conditional_vocab: bool = True,
                  wording_sigma: float | None = None) -> dict:
    """Release the DP profile of the private rows.

    Args:
        rows: private rows as {column: value}.
        row_texts: the same rows as serialized for the LLM (for length and line count).
        columns: column names in order (public schema).
        sigma: noise multiplier of each query (n_queries(conditional_vocab) of them).
        delta_s: total failure probability of the three thresholds (split equally).
        max_items: items per row for the structure family.
        max_vocab: items per row for the vocabulary family and the value wording
            (randomly kept when exceeded).
        joint_fields: also release the combination of all short values of a row.
        conditional_vocab: release the wording typical of each short column value.
        wording_sigma: noise multiplier of that query when it has a budget of its own
            (None: it shares `sigma` with the other queries).
    """
    if max_items < 1 or max_vocab < 1:
        raise ValueError("profile_max_items and profile_max_vocab must be >= 1")
    pick, noise = dp_sys_random(), dp_rng()
    # per-row family: one item per column, the combination of short values, the template
    caps = {"structure": max_items, "fields": len(columns) + 2, "vocabulary": max_vocab}
    families = {name: [] for name in caps}
    for row in rows:
        s, f, v = row_items(row, joint_fields)
        families["structure"].append(s)
        families["fields"].append(f)
        families["vocabulary"].append(v)

    released, thresholds = {}, {}
    for name, sets in families.items():
        released[name], thresholds[name] = _release(sets, caps[name], sigma, delta_s / 3, noise, pick)
    wording, wording_std, word_counts = {}, None, None
    if conditional_vocab:
        wording, wording_std, word_counts = _value_wording(
            families["fields"], families["vocabulary"], released["fields"],
            released["vocabulary"], sigma if wording_sigma is None else wording_sigma,
            max_vocab, noise, pick)
    del families

    # shape: rows per (length bin, line-count bin); sensitivity 1
    shape = np.zeros((len(LENGTH_EDGES), len(LINE_EDGES)))
    for t in row_texts:
        shape[_bin(len(t), LENGTH_EDGES), _bin(t.count("\n") + 1, LINE_EDGES)] += 1
    shape = shape + noise.normal(0.0, sigma, shape.shape)
    n_hat = max(1.0, float(shape.sum()))  # unclipped sum: unbiased row-count estimate
    # post-processing: cells below two noise standard deviations are treated as empty
    shape = np.where(shape >= 2.0 * sigma, np.clip(shape, 0.0, None), 0.0)

    def share(c):
        return float(min(1.0, max(0.0, c / n_hat)))

    profile = {"columns": list(columns), "n_records": round(n_hat),
               "n_queries": n_queries(conditional_vocab, wording_sigma is not None),
               "release_thresholds": thresholds, "item_caps": caps}
    by_kind: dict[str, list] = {k: [] for k in ("label", "symbol", "word", "phrase", "template")}
    fields: dict[str, list] = {}
    combos: list[dict] = []
    for fam in ("structure", "fields", "vocabulary"):
        for (kind, item), c in released[fam].items():
            if kind == "field":
                col, value = item.split("=", 1)
                fields.setdefault(col, []).append({"value": value, "share": share(c)})
            elif kind == "combo":
                combos.append({"values": dict(json.loads(item)), "share": share(c)})
            else:
                by_kind[kind].append({"item": item, "share": share(c)})
    for kind, entries in by_kind.items():
        profile[kind] = sorted(entries, key=lambda d: -d["share"])
    profile["fields"] = {c: sorted(v, key=lambda d: -d["share"]) for c, v in fields.items()}
    profile["combos"] = sorted(combos, key=lambda d: -d["share"])
    profile["value_wording"] = wording
    if wording_std is not None:
        profile["value_wording_noise_std"] = wording_std
        profile["value_word_counts"] = word_counts
    total = float(shape.sum()) or 1.0
    cells = []
    for i in range(len(LENGTH_EDGES)):
        for j in range(len(LINE_EDGES)):
            if shape[i, j] > 0:
                cells.append({
                    "length_min": LENGTH_EDGES[i],
                    "length_max": LENGTH_EDGES[i + 1] - 1 if i + 1 < len(LENGTH_EDGES) else None,
                    "lines_min": LINE_EDGES[j],
                    "lines_max": LINE_EDGES[j + 1] - 1 if j + 1 < len(LINE_EDGES) else None,
                    "share": float(shape[i, j] / total),
                })
    profile["shape"] = sorted(cells, key=lambda d: -d["share"])
    return profile


def render_profile(profile: dict, max_per_kind: int = 100) -> str:
    """The profile as prompt text: which conventions and words exist, most common first.

    No shares are shown. A model writing one record cannot follow a percentage, so every
    quantity with a share is drawn per row by sample_target() and stated in <Target>.
    Word shares are only a ranking (the vocabulary is subsampled per row)."""
    lines = [
        "This profile was computed from the private dataset under differential privacy. It lists",
        "the conventions and vocabulary of the real records; no real record is shown to you.",
        "Columns (in order): " + ", ".join(profile["columns"]),
    ]
    for kind, label in PROFILE_LABELS.items():
        entries = profile.get(kind) or []
        if entries:
            lines.append(f"{label} (most common first): " + ", ".join(e["item"] for e in entries[:max_per_kind]))
    if not profile.get("hide_fields"):
        for col, values in (profile.get("fields") or {}).items():
            lines.append(f"Values that occur in column {col}: " + ", ".join(f'"{v["value"]}"' for v in values[:max_per_kind]))
    return "\n".join(lines)


def _word_tables(profile: dict) -> dict:
    """log p(w | c = v) - log p(w | c) per column value, and log p(w), from the released
    noisy counts (post-processing; +1 smoothing after clipping at 0). Cached on the profile."""
    cache = profile.get("_word_tables")
    if cache is not None:
        return cache
    wc = profile["value_word_counts"]
    terms, bases = {}, []
    for col, by_val in wc["values"].items():
        vals = list(by_val)
        m = np.clip(np.asarray([by_val[v] for v in vals], dtype=float), 0.0, None) + 1.0
        base = m.sum(axis=0)
        base = base / base.sum()
        bases.append(base)
        for i, v in enumerate(vals):
            terms[(col, v)] = np.log(m[i] / m[i].sum()) - np.log(base)
    cache = {"items": [it[1] for it in wc["items"]], "terms": terms,
             "log_base": np.log(np.mean(bases, axis=0))}
    profile["_word_tables"] = cache
    return cache


def _draw_words(profile: dict, fields: dict, rng, k: int) -> list[str]:
    """k vocabulary items for one row, drawn without replacement with
    log-weight log p(w) + sum_c [log p(w | c = value_c) - log p(w | c)] (naive Bayes)."""
    t = _word_tables(profile)
    score = t["log_base"].copy()
    for col, value in fields.items():
        term = t["terms"].get((col, value))
        if term is not None:
            score = score + term
    gumbel = np.random.default_rng(rng.getrandbits(64)).gumbel(size=score.shape)
    return [t["items"][i] for i in np.argsort(-(score + gumbel))[:k]]


def _draw(values: list[dict], rng):
    """One entry drawn by share. Entries covering at least 90% of the rows are renormalized
    (the gap is noise); otherwise the remaining mass means "no target"."""
    mass = sum(v["share"] for v in values)
    if mass <= 0:
        return None
    r, acc = rng.random() * (mass if mass >= 0.9 else 1.0), 0.0
    for v in values:
        acc += v["share"]
        if r <= acc:
            return v
    return None


def _residual_fields(profile: dict) -> dict:
    """Per-column value shares among the rows NOT covered by a released combination:
    P(c = v | no combination) = (P(c = v) - P(c = v, combination)) / (1 - P(combination))."""
    combos = profile.get("combos") or []
    mass = min(sum(c["share"] for c in combos), 0.99)
    out = {}
    for col, values in (profile.get("fields") or {}).items():
        res = []
        for v in values:
            in_combos = sum(c["share"] for c in combos if c["values"].get(col) == v["value"])
            res.append({"value": v["value"], "share": max(0.0, v["share"] - in_combos) / (1.0 - mass)})
        out[col] = res if sum(r["share"] for r in res) > 0 else values
    return out


def sample_target(profile: dict, rng) -> dict:
    """Draw for this one row everything that has a share: length and line count (jointly),
    one whole label sequence (a template seen in many real rows, so the labels always form
    a combination that really occurs), the symbol patterns, and the short column values:
    one released combination of all of them (so values that go together stay together),
    or, for the share no combination covers, each column from its remaining share.
    Rare templates and combinations are not released; their share is drawn as described.
    Post-processing of the released shares only."""
    target = {"shape": None, "labels": None, "symbols": None, "fields": {}, "wording": []}
    cell = _draw(profile.get("shape") or [], rng)
    if cell is not None:
        target["shape"] = cell
    template = _draw(profile.get("template") or [], rng)
    if template is not None:
        item = template["item"]
        target["labels"] = [] if item == NO_LABELS else item.split(" → ")
    # only near-universal symbol conventions (e.g. a de-identification placeholder) are
    # required; drawing the others independently could contradict the drawn template
    target["symbols"] = [e["item"] for e in profile.get("symbol") or [] if e["share"] >= 0.9]
    combo = _draw(profile.get("combos") or [], rng)
    if combo is not None:
        target["fields"] = dict(combo["values"])
    else:
        for col, values in _residual_fields(profile).items():
            v = _draw(values, rng)
            if v is not None:
                target["fields"][col] = v["value"]
    wording = profile.get("value_wording") or {}
    for col, value in target["fields"].items():
        items = wording.get(col, {}).get(value)
        if items:
            target["wording"].append((col, value, [e["item"] for e in items]))
    k = int(profile.get("words_per_row") or 0)
    if k > 0 and profile.get("value_word_counts"):
        target["words"] = _draw_words(profile, target["fields"], rng, k)
    # the values are drawn (they become the row's values) but not shown to the model
    target["hide_fields"] = bool(profile.get("hide_fields"))
    return target


def _span(lo, hi):
    if hi is None:
        return f"{lo:,} or more"
    return f"{lo:,}" if hi == lo else f"{lo:,}-{hi:,}"


def render_target(target: dict) -> str:
    lines = ["Targets drawn for THIS row from the real distribution. Follow every one exactly:"]
    cell = target.get("shape")
    if cell:
        lines.append(f"- Length of the whole CSV row: {_span(cell['length_min'], cell['length_max'])} characters")
        lines.append(f"- Number of lines: {_span(cell['lines_min'], cell['lines_max'])}")
    if target.get("labels") is not None:
        if target["labels"]:
            lines.append("- Labels (name followed by a colon), exactly these and in this order: "
                         + ", ".join(target["labels"]))
        else:
            lines.append("- Use none of the labels listed in the profile")
    if target.get("symbols"):
        lines.append("- Use these symbol patterns the way the data does: " + ", ".join(target["symbols"]))
    if target.get("words"):
        lines.append("- Words and phrases drawn for THIS record from the real data; use every one "
                     "of them, each where it fits naturally: " + ", ".join(target["words"]))
    if not target.get("hide_fields"):
        for col, value in target.get("fields", {}).items():
            lines.append(f'- Column {col} = "{value}"')
        for col, value, items in target.get("wording") or []:
            lines.append(f'- Wording that real records with {col} = "{value}" use clearly more often '
                         "than other records: " + ", ".join(items))
    lines.append("Anything not fixed here is free: choose it to fit the profile and these targets.")
    return "\n".join(lines)
