#!/usr/bin/env python3
"""Convert a column type document (a Markdown table) into the column_spec file used by the DPflow_* synthesizers.

  python -m petsard.synthesizer.DPflow_column_spec <type document .md> -o column_spec.yaml
  python -m petsard.synthesizer.DPflow_column_spec <type document .md> -o column_spec.yaml --names names.txt

The table needs the columns 欄位名稱 (name), 型別 (type) and 可能值 (values); 序號 (no) and 整數 (integer) are
optional. English headers name, type, values, no and integer are accepted too. Each row is checked: names unique and
in order, type 類別/cat or 數值/num, possible values for every categorical column, the integer flag only on numeric
columns. With --names (a JSON list, or a text file with one name per line) the names and their order are also checked
against that list.

The document should come from the data's public definition (a data dictionary or a schema agreed with the data
owner), not from statistics of the private data. This program reads only the document and the optional name list;
no data file is opened. Any problem stops the program with exit code 1 and lists the rows concerned.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import yaml

TYPES = {"類別": "cat", "類別型": "cat", "cat": "cat", "categorical": "cat", "category": "cat",
         "數值": "num", "數值型": "num", "num": "num", "numeric": "num", "numerical": "num"}
YES = {"是", "yes", "y", "true"}
NO = {"否", "no", "n", "false", "-", ""}
EMPTY = {"", "-"}
HEADERS = {"序號": ("序號", "no", "#"), "欄位名稱": ("欄位名稱", "name", "column"), "型別": ("型別", "type"),
           "可能值": ("可能值", "values"), "整數": ("整數", "integer")}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip()
    if len(s) >= 2 and s[0] == s[-1] == "`":
        s = s[1:-1].strip()
    return s


def cells(line: str) -> list[str]:
    body = line.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") and not body.endswith("\\|") else body
    return [norm(c.replace("\\|", "|")) for c in re.split(r"(?<!\\)\|", body)]


def parse_values(text: str) -> list:
    parts = [p.strip() for p in re.split(r"[,，、;；]", unicodedata.normalize("NFKC", text))]
    parts = [p for p in parts if p]
    try:
        return [float(p) for p in parts]
    except ValueError:
        return parts


def read_report(path: Path) -> tuple[list[dict], list[str], dict]:
    lines = path.read_text(encoding="utf-8-sig").replace("\r", "").split("\n")
    errors, info = [], {}
    def header_index(row: list[str]) -> dict:
        low = [c.lower() for c in row]
        return {key: low.index(alias) for key, aliases in HEADERS.items() for alias in aliases if alias in low}

    head = next((i for i, ln in enumerate(lines) if ln.strip().startswith("|")
                 and {"欄位名稱", "型別"} <= set(header_index(cells(ln)))), None)
    if head is None:
        return [], ["找不到含「欄位名稱」(name)與「型別」(type)的表頭。"], info
    names = cells(lines[head])
    idx = header_index(names)
    for need in ("欄位名稱", "型別", "可能值"):
        if need not in idx:
            errors.append(f"表頭缺少「{need}」欄。")
    if errors:
        return [], errors, info

    spec, seen = [], set()
    for no, ln in enumerate(lines[head + 1:], head + 2):
        if not ln.strip().startswith("|"):
            break
        row = cells(ln)
        if all(re.fullmatch(r":?-{3,}:?", c) for c in row if c):
            continue
        if len(row) != len(names):
            errors.append(f"第{no}行：欄數{len(row)}與表頭{len(names)}不同。")
            continue
        name, kind_text = row[idx["欄位名稱"]], row[idx["型別"]]
        where = f"第{no}行({name or '無名稱'})"
        if "序號" in idx and row[idx["序號"]] != str(len(spec) + 1):
            errors.append(f"{where}：序號應為{len(spec) + 1}，表上是「{row[idx['序號']]}」，可能有列被刪除或調換。")
        if not name:
            errors.append(f"{where}：欄位名稱是空的。")
        elif name in seen:
            errors.append(f"{where}：欄位名稱重複。")
        seen.add(name)
        kind = TYPES.get(kind_text.lower())
        if kind is None:
            errors.append(f"{where}：型別「{kind_text}」只能是「類別」或「數值」。")
            kind = "num"
        values_text = row[idx["可能值"]]
        integer_text = row[idx["整數"]].lower() if "整數" in idx else ""
        if integer_text not in YES | NO:
            errors.append(f"{where}：整數「{integer_text}」只能是「是」、「否」或「-」。")
        entry = {"name": name, "type": kind}
        if kind == "cat":
            values = [] if values_text in EMPTY else parse_values(values_text)
            if not values:
                errors.append(f"{where}：類別欄要列出可能值。")
            elif len(set(values)) != len(values):
                errors.append(f"{where}：可能值重複。")
            if integer_text in YES:
                errors.append(f"{where}：整數只用於數值欄，類別欄的整數應為「-」。")
            entry["values"] = values
        else:
            if values_text not in EMPTY:
                errors.append(f"{where}：數值欄的可能值應為「-」，表上是「{values_text}」。")
            if integer_text in YES:
                entry["integer"] = True
        spec.append(entry)

    title = next((ln.lstrip("#").strip() for ln in lines if ln.startswith("# ")), "")
    if title:
        info["文件"] = title
    return spec, errors, info


def check_names(spec: list[dict], names_file: Path) -> list[str]:
    """Compare the declared column names and their order with a list: JSON list, or one name per line."""
    text = names_file.read_text(encoding="utf-8-sig")
    try:
        want = [str(n) for n in json.loads(text)]
    except json.JSONDecodeError:
        want = [ln.strip() for ln in text.splitlines() if ln.strip()]
    have = [c["name"] for c in spec]
    if have == want:
        return []
    errors = [f"欄位與名稱清單不一致：表上{len(have)}欄，清單{len(want)}欄。"]
    missing, extra = [n for n in want if n not in have], [n for n in have if n not in want]
    if missing:
        errors.append(f"缺少：{missing[:10]}{' ...' if len(missing) > 10 else ''}")
    if extra:
        errors.append(f"多出：{extra[:10]}{' ...' if len(extra) > 10 else ''}")
    if not missing and not extra:
        first = next(i for i, (a, b) in enumerate(zip(have, want)) if a != b)
        errors.append(f"順序不同，第{first + 1}欄表上是{have[first]}，應為{want[first]}。")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a column type document (.md table) to column_spec.yaml")
    ap.add_argument("report", help="column type document (.md)")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--names", help="expected column names in order: a JSON list or one name per line")
    a = ap.parse_args()

    spec, errors, info = read_report(Path(a.report))
    if not spec and not errors:
        errors.append("表格沒有任何欄位。")
    if a.names and spec:
        errors += check_names(spec, Path(a.names))
    if errors:
        print(f"未產生{a.out}，共{len(errors)}個問題：", file=sys.stderr)
        for e in errors[:50]:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    Path(a.out).write_text(yaml.safe_dump({"columns": spec}, allow_unicode=True, sort_keys=False), encoding="utf-8")
    n_cat = sum(c["type"] == "cat" for c in spec)
    n_int = sum(bool(c.get("integer")) for c in spec)
    print(f"wrote {a.out}: {len(spec)} columns, cat {n_cat}, num {len(spec) - n_cat}, integer {n_int}")
    for k, v in info.items():
        print(f"  {k}：{v}")


if __name__ == "__main__":
    main()
