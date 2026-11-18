#!/usr/bin/env python3
"""
AG Ruler (ag_ruler.py)
LLM（トークナイザー）の死角となる物理フォーマット（BOM・文字コード・改行・不可視文字）と
主要構造ファイルの構文をゼロ依存（Python標準ライブラリのみ）で検査する決定論的定規CLI。
"""

from __future__ import annotations

import argparse
import ast
import bisect
import codecs
import csv
import io
import json
import re
import sys
import tokenize
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11 fallback
    tomllib = None

# 不可視文字・地雷文字の既知ラベル定義
HAZARD_CHARS: dict[str, tuple[str, str]] = {
    "\u00a0": ("U+00A0", "NBSP (No-Break Space)"),
    "\u200b": ("U+200B", "ZWSP (Zero-Width Space)"),
    "\u200c": ("U+200C", "ZWNJ (Zero-Width Non-Joiner)"),
    "\u200d": ("U+200D", "ZWJ (Zero-Width Joiner)"),
    "\ufeff": ("U+FEFF", "In-body BOM / ZWNBSP"),
    "\ufffd": ("U+FFFD", "Replacement Character (文字化け痕跡)"),
    "\u2060": ("U+2060", "WORD JOINER"),
    "\u00ad": ("U+00AD", "SOFT HYPHEN"),
    "\u034f": ("U+034F", "COMBINING GRAPHEME JOINER"),
    "\u115f": ("U+115F", "HANGUL CHOSEONG FILLER"),
    "\u1160": ("U+1160", "HANGUL JUNGSEONG FILLER"),
    "\u2800": ("U+2800", "BRAILLE PATTERN BLANK"),
    "\u3164": ("U+3164", "HANGUL FILLER"),
    "\uffa0": ("U+FFA0", "HALFWIDTH HANGUL FILLER"),
}

EXT_TO_TYPE: dict[str, str] = {
    ".py": "py",
    ".json": "json",
    ".jsonc": "jsonc",
    ".toml": "toml",
    ".csv": "csv",
    ".tsv": "tsv",
    ".xml": "xml",
    ".svg": "xml",
    ".yaml": "yaml",
    ".yml": "yml",
    ".sh": "sh",
    ".ps1": "ps1",
    ".js": "js",
    ".ts": "ts",
    ".md": "md",
    ".txt": "text",
}

INDENT_CHECK_TYPES = set(EXT_TO_TYPE.values()) - {"md", "text", "csv", "tsv", "xml"}
# パーサーが構文位置の全角スペースを弾けない（またはパーサーを持たない）種別
FULLWIDTH_SPACE_SCAN_TYPES = {"yaml", "yml", "sh", "ps1", "js", "ts"}
EOL_PATTERN = re.compile(r"\r\n|\r|\n")

_NON_CODE_TOKENS = {tokenize.STRING, tokenize.COMMENT}
if hasattr(tokenize, "FSTRING_MIDDLE"):
    _NON_CODE_TOKENS.add(tokenize.FSTRING_MIDDLE)

_RGI_TAG_FLAGS = re.compile(
    "\U0001F3F4(?:"
    + "|".join("".join(chr(0xE0000 + ord(c)) for c in code) for code in ("gbeng", "gbsct", "gbwls"))
    + ")\U000E007F"
)

MAX_FINDINGS = 15


def _infer_file_type(p: Path, forced_type: str | None) -> str:
    if forced_type:
        return forced_type
    name_lower = p.name.lower()
    if name_lower in ("tsconfig.json", "jsconfig.json") or (
        p.suffix.lower() == ".json" and ".vscode" in [part.lower() for part in p.parts]
    ):
        return "jsonc"
    return EXT_TO_TYPE.get(p.suffix.lower(), "text")


def count_physical_lines(text: str) -> int:
    """エディタ基準の物理行数を返す（末尾改行の終端は空行として数えない）。"""
    if not text:
        return 0
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return len(lines)


def detect_bom(raw: bytes) -> tuple[str | None, bytes]:
    """BOMを検出し、(BOM種別名, BOM除去後バイト列) を返す。"""
    boms = [
        (codecs.BOM_UTF32_LE, "UTF-32-LE-BOM"),
        (codecs.BOM_UTF32_BE, "UTF-32-BE-BOM"),
        (codecs.BOM_UTF8, "UTF-8-BOM"),
        (codecs.BOM_UTF16_LE, "UTF-16-LE-BOM"),
        (codecs.BOM_UTF16_BE, "UTF-16-BE-BOM"),
    ]
    for bom_bytes, bom_name in boms:
        if raw.startswith(bom_bytes):
            return bom_name, raw[len(bom_bytes):]
    return None, raw


def detect_line_endings(text: str) -> tuple[str, dict]:
    """デコード済みテキストから改行コード(LF / CRLF / CR / MIXED / NONE)と行番号詳細を判定する。"""
    if not text:
        return "NONE", {"lf": 0, "crlf": 0, "cr": 0, "mixed_samples": []}

    lf_lines: list[int] = []
    crlf_lines: list[int] = []
    cr_lines: list[int] = []

    for idx, m in enumerate(EOL_PATTERN.finditer(text), start=1):
        sep = m.group(0)
        if sep == "\r\n":
            crlf_lines.append(idx)
        elif sep == "\n":
            lf_lines.append(idx)
        else:
            cr_lines.append(idx)

    counts = {"LF": len(lf_lines), "CRLF": len(crlf_lines), "CR": len(cr_lines)}
    active = {k: v for k, v in counts.items() if v > 0}

    if not active:
        return "NONE", {"lf": 0, "crlf": 0, "cr": 0, "mixed_samples": []}
    if len(active) == 1:
        only_kind = next(iter(active))
        return only_kind, {"lf": len(lf_lines), "crlf": len(crlf_lines), "cr": len(cr_lines), "mixed_samples": []}

    majority = max(active, key=lambda k: active[k])
    samples: list[str] = []
    if majority != "CRLF" and crlf_lines:
        samples.append(f"CRLF@{crlf_lines[:5]}")
    if majority != "LF" and lf_lines:
        samples.append(f"LF@{lf_lines[:5]}")
    if majority != "CR" and cr_lines:
        samples.append(f"CR@{cr_lines[:5]}")

    summary = f"MIXED({','.join(f'{k}:{v}' for k, v in active.items())})"
    return summary, {
        "lf": len(lf_lines),
        "crlf": len(crlf_lines),
        "cr": len(cr_lines),
        "mixed_samples": samples,
    }


def decode_text(raw_body: bytes, bom_name: str | None) -> tuple[str | None, str, str | None]:
    """バイト列をテキストへデコードする。(text, encoding_label, error_msg) を返す。"""
    target_enc = "utf-8"
    if bom_name == "UTF-16-LE-BOM":
        target_enc = "utf-16-le"
    elif bom_name == "UTF-16-BE-BOM":
        target_enc = "utf-16-be"
    elif bom_name == "UTF-32-LE-BOM":
        target_enc = "utf-32-le"
    elif bom_name == "UTF-32-BE-BOM":
        target_enc = "utf-32-be"

    try:
        text = raw_body.decode(target_enc, errors="strict")
        label = bom_name if bom_name else "UTF-8"
        return text, label, None
    except UnicodeDecodeError as e:
        line_no = raw_body[:e.start].count(b"\n") + 1
        hint = ""
        try:
            raw_body.decode("cp932", errors="strict")
            hint = " [decodable as cp932/Shift-JIS]"
        except UnicodeDecodeError:
            pass
        return None, f"INVALID-{target_enc.upper()}", (
            f"Decode error ({target_enc}) at byte {e.start} (line {line_no}): {e.reason}{hint}"
        )


def _python_non_code_spans(text: str) -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    """Pythonソースから文字列・コメント・f-stringリテラルの(start, end)位置を収集する。"""
    spans: list[tuple[tuple[int, int], tuple[int, int]]] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in _NON_CODE_TOKENS:
                spans.append((tok.start, tok.end))
    except (tokenize.TokenError, SyntaxError, ValueError):
        pass
    starts = [s for s, _ in spans]
    return starts, spans


def _in_spans(
    line_no: int,
    col_idx0: int,
    non_code: tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]]],
) -> bool:
    starts, spans = non_code
    if not starts:
        return False
    pos = (line_no, col_idx0)
    i = bisect.bisect_right(starts, pos) - 1
    return i >= 0 and pos < spans[i][1]


def _is_emoji_like(ch: str) -> bool:
    o = ord(ch)
    return 0x1F000 <= o <= 0x1FAFF or 0x2190 <= o <= 0x2BFF or o == 0xFE0F


def _is_emoji_zwj(line: str, idx: int) -> bool:
    """md/text内で前後が絵文字域に挟まれたZWJかを判定する。"""
    return (
        idx > 0
        and _is_emoji_like(line[idx - 1])
        and idx + 1 < len(line)
        and _is_emoji_like(line[idx + 1])
    )


def _is_variation_selector(ch: str) -> bool:
    o = ord(ch)
    return 0xFE00 <= o <= 0xFE0F or 0xE0100 <= o <= 0xE01EF


def _is_cjk_ideograph(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF
        or 0x3400 <= o <= 0x4DBF
        or 0xF900 <= o <= 0xFAFF
        or 0x20000 <= o <= 0x2FFFF
    )


def _is_valid_vs_base(prev_ch: str, vs_ch: str) -> bool:
    o = ord(vs_ch)
    if 0xE0100 <= o <= 0xE01EF:
        return _is_cjk_ideograph(prev_ch)
    if o in (0xFE0E, 0xFE0F):
        po = ord(prev_ch)
        return (
            _is_emoji_like(prev_ch)
            or prev_ch.isdigit()
            or prev_ch in "#*©®™‼⁉"
            or 0x203C <= po <= 0x3299
        )
    return False


def _in_char_spans(idx: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= idx < end for start, end in spans)


def _describe_hazard_char(ch: str) -> tuple[str, str] | None:
    """検査対象の不可視・制御・地雷文字なら (表示ラベル, level) を返す。"""
    if ch in HAZARD_CHARS:
        code_pt, desc = HAZARD_CHARS[ch]
        level = "ERR" if ch == "\ufffd" else "WARN"
        return f"{code_pt} ({desc})", level

    o = ord(ch)
    if 0xE0000 <= o <= 0xE007F:
        return "U+E0000-E007F (Unicode Tag Char / ASCII Smuggling)", "ERR"

    cat = unicodedata.category(ch)
    if cat in ("Cc", "Cf", "Co") and ch not in ("\t", "\n", "\r"):
        name = unicodedata.name(ch, "Control/Format Char")
        level = "ERR" if cat == "Cc" else "WARN"
        return f"U+{o:04X} ({name})", level

    if cat in ("Zl", "Zp") or (cat == "Zs" and ch not in (" ", "\u3000")):
        name = unicodedata.name(ch, "Unicode Space/Separator")
        return f"U+{o:04X} ({name})", "WARN"

    return None


def scan_text_hazards(text: str, file_type: str) -> list[dict]:
    """不可視文字・制御文字・コード内全角スペース・インデント混在を検出する。"""
    findings: list[dict] = []
    char_hits: dict[str, tuple[str, list[str]]] = {}
    tab_indent_lines: list[int] = []
    space_indent_lines: list[int] = []

    check_indent = file_type in INDENT_CHECK_TYPES
    py_non_code = _python_non_code_spans(text) if file_type == "py" else ([], [])

    raw_lines = text.split("\n")
    if raw_lines and raw_lines[-1] == "":
        raw_lines.pop()

    for line_no, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.rstrip("\r")
        flag_spans = [m.span() for m in _RGI_TAG_FLAGS.finditer(line)]
        if check_indent and line:
            if line.startswith("\t"):
                tab_indent_lines.append(line_no)
            elif line.startswith(" "):
                space_indent_lines.append(line_no)

        for idx, ch in enumerate(line):
            col_no = idx + 1
            if file_type == "py" and idx == 0 and ch == "\x0c":
                continue

            if ch == "\u200d" and file_type in ("md", "text") and _is_emoji_zwj(line, idx):
                continue

            if 0xE0000 <= ord(ch) <= 0xE007F and _in_char_spans(idx, flag_spans):
                continue

            if _is_variation_selector(ch):
                if idx > 0 and _is_variation_selector(line[idx - 1]):
                    key = "U+FE00..FE0F/E0100..E01EF (Consecutive Variation Selectors)"
                    entry = char_hits.setdefault(key, ("ERR", []))
                    entry[1].append(f"{line_no}:{col_no}")
                elif idx == 0 or not _is_valid_vs_base(line[idx - 1], ch):
                    key = f"U+{ord(ch):04X} (Unexpected Variation Selector)"
                    entry = char_hits.setdefault(key, ("WARN", []))
                    entry[1].append(f"{line_no}:{col_no}")
                continue

            if ch == "\u3000":
                if file_type in FULLWIDTH_SPACE_SCAN_TYPES or (
                    file_type == "py" and not _in_spans(line_no, idx, py_non_code)
                ):
                    key = "U+3000 (Full-width Space in code/config)"
                    entry = char_hits.setdefault(key, ("WARN", []))
                    entry[1].append(f"{line_no}:{col_no}")
                continue

            described = _describe_hazard_char(ch)
            if described:
                key, level = described
                entry = char_hits.setdefault(key, (level, []))
                entry[1].append(f"{line_no}:{col_no}")

    for key, (level, locs) in char_hits.items():
        findings.append({
            "level": level,
            "kind": "CHAR",
            "message": f"{key} x{len(locs)} at {locs[:5]}",
        })

    if file_type in ("yaml", "yml") and tab_indent_lines:
        findings.append({
            "level": "ERR",
            "kind": "INDENT",
            "message": f"YAML prohibits tab indentation (Tabs@{tab_indent_lines[:5]})",
        })
    elif check_indent and tab_indent_lines and space_indent_lines:
        findings.append({
            "level": "WARN",
            "kind": "INDENT",
            "message": f"Mixed Tab/Space indentation (Tabs@{tab_indent_lines[:3]}, Spaces@{space_indent_lines[:3]})",
        })

    return findings


def _check_bom_and_eol_policy(
    file_type: str,
    bom_name: str | None,
    text: str,
    eol_summary: str,
    eol_info: dict,
) -> list[dict]:
    """種別ごとのBOM・改行ポリシーを評価する。"""
    findings: list[dict] = []

    if bom_name:
        if file_type in ("json", "jsonc", "toml", "sh"):
            level: str | None = "ERR"
        elif file_type == "py":
            level = "ERR" if text.startswith("#!") else "WARN"
        elif file_type == "ps1" and bom_name == "UTF-8-BOM":
            level = None  # Windows PowerShell 5.1 互換のため許可
        elif file_type in ("csv", "tsv") and bom_name == "UTF-8-BOM":
            level = "INFO"  # 日本語版 Excel 向け正当用途あり
        else:
            level = "WARN"

        if level:
            findings.append({"level": level, "kind": "BOM", "message": f"{bom_name} detected"})
    elif file_type == "ps1" and any(ord(ch) > 127 for ch in text):
        findings.append({
            "level": "WARN",
            "kind": "BOM",
            "message": "Non-ASCII .ps1 without UTF-8-BOM (may misdecode as cp932 in Windows PowerShell 5.1)",
        })

    has_crlf = eol_summary == "CRLF" or eol_info.get("crlf", 0) > 0
    if (file_type == "sh" or (file_type == "py" and text.startswith("#!"))) and has_crlf:
        detail = f"{eol_summary} {eol_info['mixed_samples']}".strip() if eol_info["mixed_samples"] else eol_summary
        findings.append({
            "level": "ERR",
            "kind": "EOL",
            "message": f"CRLF in {file_type} script breaks shebang / interpreter ({detail})",
        })
    elif eol_summary.startswith("MIXED"):
        findings.append({
            "level": "WARN",
            "kind": "EOL",
            "message": f"Mixed line endings: {', '.join(eol_info['mixed_samples'])}",
        })

    return findings


def _json_pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
    seen: set[str] = set()
    dups: list[str] = []
    out: dict[str, object] = {}
    for k, v in pairs:
        if k in seen:
            dups.append(k)
        seen.add(k)
        out[k] = v
    if dups:
        raise ValueError(f"Duplicate JSON key(s): {dups[:5]}")
    return out


def _reject_json_constant(val: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {val}")


def _strip_jsonc(text: str) -> str:
    """JSONCのコメント(//, /* */)と末尾カンマを文字列リテラルを壊さずに除去する。"""
    out: list[str] = []
    i = 0
    n = len(text)
    in_str = False
    last_sig = -1
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
                last_sig = len(out) - 1
            i += 1
        elif ch == '"':
            in_str = True
            out.append(ch)
            i += 1
        elif ch == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] not in ("\r", "\n"):
                i += 1
        elif ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            closed = False
            while i + 1 < n:
                if text[i] == "*" and text[i + 1] == "/":
                    i += 2
                    closed = True
                    break
                if text[i] == "\n":
                    out.append("\n")
                i += 1
            if not closed:
                raise ValueError("Unterminated block comment")
        else:
            if ch in "}]" and last_sig >= 0 and out[last_sig] == ",":
                out[last_sig] = ""
            out.append(ch)
            if not ch.isspace():
                last_sig = len(out) - 1
            i += 1
    return "".join(out)


def validate_syntax(text: str, file_type: str, bom_name: str | None = None) -> tuple[str, list[dict]]:
    """拡張子に応じた軽量構文バリデーションを実行する。(status_tag, findings) を返す。"""
    findings: list[dict] = []

    if file_type in ("json", "jsonc"):
        raw_json_text = ("\ufeff" + text) if bom_name == "UTF-8-BOM" else text
        try:
            if file_type == "jsonc":
                raw_json_text = _strip_jsonc(raw_json_text)
            json.loads(
                raw_json_text,
                object_pairs_hook=_json_pairs_hook,
                parse_constant=_reject_json_constant,
            )
            return f"{file_type}:valid", findings
        except json.JSONDecodeError as e:
            findings.append({
                "level": "ERR",
                "kind": "SYNTAX",
                "message": f"{file_type.upper()} error at line {e.lineno}, col {e.colno}: {e.msg}",
            })
            return f"{file_type}:ERROR", findings
        except (ValueError, RecursionError) as e:
            findings.append({"level": "ERR", "kind": "SYNTAX", "message": f"{file_type.upper()} error: {e}"})
            return f"{file_type}:ERROR", findings

    if file_type == "toml":
        if tomllib is None:
            return "toml:skipped(py<3.11)", findings
        raw_toml_text = ("\ufeff" + text) if bom_name == "UTF-8-BOM" else text
        try:
            tomllib.loads(raw_toml_text)
            return "toml:valid", findings
        except tomllib.TOMLDecodeError as e:
            findings.append({
                "level": "ERR",
                "kind": "SYNTAX",
                "message": f"TOML error: {e}",
            })
            return "toml:ERROR", findings

    if file_type == "py":
        try:
            ast.parse(text)
            return "py:valid", findings
        except SyntaxError as e:
            findings.append({
                "level": "ERR",
                "kind": "SYNTAX",
                "message": f"Python SyntaxError at line {e.lineno}, col {e.offset}: {e.msg}",
            })
            return "py:ERROR", findings
        except (ValueError, RecursionError) as e:
            findings.append({"level": "ERR", "kind": "SYNTAX", "message": f"Python parse error: {e}"})
            return "py:ERROR", findings

    if file_type in ("csv", "tsv"):
        delimiter = "\t" if file_type == "tsv" else ","
        try:
            reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
            header = next(reader, None)
            if header is None:
                return f"{file_type}:empty", findings
            expected_cols = len(header)
            row_count = 1
            mismatch_lines: list[str] = []
            for row in reader:
                row_count += 1
                if len(row) != expected_cols:
                    desc = "blank line" if len(row) == 0 else f"cols={len(row)}"
                    mismatch_lines.append(f"{reader.line_num}({desc})")
            if mismatch_lines:
                findings.append({
                    "level": "WARN",
                    "kind": "SYNTAX",
                    "message": f"{file_type.upper()} column count mismatch (header={expected_cols}) at lines {mismatch_lines[:5]}",
                })
                return f"{file_type}:WARN", findings
            return f"{file_type}:valid({row_count}rows,{expected_cols}cols)", findings
        except csv.Error as e:
            findings.append({"level": "ERR", "kind": "SYNTAX", "message": f"{file_type.upper()} parse error: {e}"})
            return f"{file_type}:ERROR", findings

    if file_type == "xml":
        try:
            ET.fromstring(text)
            return "xml:valid", findings
        except ET.ParseError as e:
            findings.append({"level": "ERR", "kind": "SYNTAX", "message": f"XML error: {e}"})
            return "xml:ERROR", findings

    return f"{file_type}:skipped(no-validator)", findings


def _cap_findings(findings: list[dict]) -> list[dict]:
    """ERRを優先ソートし、MAX_FINDINGSを超える場合は要約行に畳む。"""
    rank = {"ERR": 0, "WARN": 1, "INFO": 2}
    ordered = sorted(findings, key=lambda f: rank.get(f["level"], 9))
    if len(ordered) <= MAX_FINDINGS:
        return ordered
    dropped = ordered[MAX_FINDINGS:]
    max_dropped_level = min(dropped, key=lambda f: rank.get(f["level"], 9))["level"]
    kept = ordered[:MAX_FINDINGS]
    kept.append({
        "level": max_dropped_level,
        "kind": "TRUNCATED",
        "message": f"... (+{len(dropped)} more findings omitted, max level={max_dropped_level})",
    })
    return kept


def inspect_file(path_str: str, forced_type: str | None = None) -> dict:
    """1ファイルを物理検査・構文検査し、構造化結果を返す。"""
    p = Path(path_str)
    if not p.exists():
        return {
            "path": path_str,
            "status": "ERR",
            "findings": [{"level": "ERR", "kind": "IO", "message": "File not found"}],
        }
    if p.is_dir():
        return {
            "path": path_str,
            "status": "ERR",
            "findings": [{"level": "ERR", "kind": "IO", "message": "Path is a directory"}],
        }

    try:
        raw = p.read_bytes()
    except OSError as e:
        return {
            "path": path_str,
            "status": "ERR",
            "findings": [{"level": "ERR", "kind": "IO", "message": f"Read error: {e}"}],
        }

    size_bytes = len(raw)
    bom_name, raw_body = detect_bom(raw)
    file_type = _infer_file_type(p, forced_type)

    text, enc_label, decode_err = decode_text(raw_body, bom_name)
    if decode_err or text is None:
        raw_lines = raw_body.count(b"\n") + (0 if raw_body.endswith(b"\n") or not raw_body else 1)
        findings = [{"level": "ERR", "kind": "ENCODING", "message": decode_err or "Decode failed"}]
        return {
            "path": path_str,
            "status": "ERR",
            "encoding": enc_label,
            "eol": "UNKNOWN",
            "lines": raw_lines,
            "bytes": size_bytes,
            "syntax": f"{file_type}:skipped(decode-error)",
            "findings": findings,
        }

    eol_summary, eol_info = detect_line_endings(text)
    line_count = count_physical_lines(text)

    findings = _check_bom_and_eol_policy(file_type, bom_name, text, eol_summary, eol_info)
    findings.extend(scan_text_hazards(text, file_type))
    syntax_tag, syntax_findings = validate_syntax(text, file_type, bom_name=bom_name)
    findings.extend(syntax_findings)

    has_err = any(f["level"] == "ERR" for f in findings)
    has_warn = any(f["level"] == "WARN" for f in findings)
    status = "ERR" if has_err else ("WARN" if has_warn else "OK")

    return {
        "path": path_str,
        "status": status,
        "encoding": enc_label,
        "eol": eol_summary,
        "lines": line_count,
        "bytes": size_bytes,
        "syntax": syntax_tag,
        "findings": _cap_findings(findings),
    }


def format_compact(res: dict) -> str:
    """AIエージェント向けの超省トークンサマリー文字列を生成する。"""
    if "encoding" not in res:
        lines = [f"{res['status']} | {res['path']}"]
    else:
        lines = [
            f"{res['status']} | {res['path']} | {res['encoding']} | {res['eol']} "
            f"| {res['lines']} lines | {res['bytes']:,} bytes | {res['syntax']}"
        ]
    for f in res["findings"]:
        lines.append(f"  - [{f['level']}:{f['kind']}] {f['message']}")
    return "\n".join(lines)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    parser = argparse.ArgumentParser(
        prog="ag-ruler",
        description="AG Ruler: AIエージェントのための物理フォーマット・文字コード・構文検査定規 (Zero Dependencies)",
        epilog="Created by: Bug-san (Tamakoro) & Antigravity | Bug reports / updates: ask Bug-san or Antigravity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", help="検査対象のファイルパス（複数指定可）")
    parser.add_argument(
        "--type",
        choices=sorted(set(EXT_TO_TYPE.values())),
        default=None,
        help="ファイル種別を強制指定（省略時は拡張子から自動判定）",
    )
    parser.add_argument("--json", action="store_true", help="JSON形式で出力する")
    parser.add_argument("--strict", action="store_true", help="WARN（警告）でも終了コード1を返す")
    args = parser.parse_args()

    results = [inspect_file(p, forced_type=args.type) for p in args.paths]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for res in results:
            print(format_compact(res))

    has_err = any(r["status"] == "ERR" for r in results)
    has_warn = any(r["status"] == "WARN" for r in results)
    if has_err or (args.strict and has_warn):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
