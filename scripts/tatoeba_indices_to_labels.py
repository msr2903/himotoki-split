#!/usr/bin/env python3
"""Convert Tatoeba Japanese indices (Tanaka B-lines) into split labels.

Tatoeba ``jpn_indices`` link words to **JMdict** (reading / seq# / sense).
Indexed tokens often omit proper nouns, numbers, and punctuation — we
**greedily align** surface forms into the sentence and treat leftover spans
as segments too.

Download:
  https://downloads.tatoeba.org/exports/jpn_indices.tar.bz2

Example:
  python scripts/tatoeba_indices_to_labels.py --download \\
    -o data/labels/tatoeba_indices.jsonl
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

DEFAULT_URL = "https://downloads.tatoeba.org/exports/jpn_indices.tar.bz2"
SENTENCES_URL = (
    "https://downloads.tatoeba.org/exports/per_language/jpn/jpn_sentences.tsv.bz2"
)
USER_AGENT = "himotoki-split-tatoeba/0.1 (https://github.com/msr2903/himotoki-split)"

_TOKEN_RE = re.compile(
    r"^"
    r"(?P<head>.+?)"
    r"(?:\((?P<reading>[^)]*)\))?"
    r"(?:\[(?P<sense>\d+)\])?"
    r"(?:\|(?P<flag>\d+))?"
    r"(?:\{(?P<surface>[^}]*)\})?"
    r"$"
)


def _urlopen(url: str, timeout: int = 180):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 50_000:
        print(f"Using existing: {dest}")
        return dest
    print(f"Downloading {url} -> {dest}")
    with _urlopen(url) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            print(f"  {dest.stat().st_size // 10**6} MB", flush=True)
    return dest


def parse_index_token(tok: str) -> Optional[str]:
    """Return surface form for one B-line token, or None to skip."""
    tok = tok.strip()
    if not tok or tok == "~":
        return None
    # Compound marker: ログアウト~
    if tok.endswith("~"):
        tok = tok[:-1]
        if not tok:
            return None
    m = _TOKEN_RE.match(tok)
    if not m:
        if "(" not in tok and "{" not in tok and "[" not in tok:
            return tok
        return None
    surface = m.group("surface")
    head = m.group("head")
    if surface:
        return surface
    if head:
        return head
    return None


def extract_surfaces(index_text: str) -> Optional[List[str]]:
    surfaces: List[str] = []
    for p in index_text.split():
        s = parse_index_token(p)
        if s is None:
            # skip pure ~ / unparsable noise; don't abort whole line
            continue
        if s:
            surfaces.append(s)
    return surfaces or None


def align_segments(text: str, surfaces: List[str]) -> Optional[List[str]]:
    """Greedy L→R align indexed surfaces into ``text``; leftovers become segments."""
    pos = 0
    segs: List[str] = []
    for surf in surfaces:
        if not surf:
            continue
        idx = text.find(surf, pos)
        if idx < 0:
            return None
        if idx > pos:
            segs.append(text[pos:idx])
        segs.append(surf)
        pos = idx + len(surf)
    if pos < len(text):
        segs.append(text[pos:])
    if not segs or "".join(segs) != text:
        return None
    # Drop empty (shouldn't happen)
    segs = [s for s in segs if s]
    return segs or None


def load_sentence_map(tsv_bz2: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with bz2.open(tsv_bz2, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sid, lang, text = parts[0], parts[1], parts[2]
            if lang == "jpn":
                out[sid] = text
    return out


def iter_index_rows(path: Path) -> Iterator[Tuple[str, str, str]]:
    if str(path).endswith(".tar.bz2"):
        with tarfile.open(path, "r:bz2") as tar:
            member = tar.getmembers()[0]
            f = tar.extractfile(member)
            assert f is not None
            for raw in f:
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                parts = line.split("\t")
                if len(parts) < 3:
                    continue
                yield parts[0], parts[1], parts[2]
    else:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                yield parts[0], parts[1], parts[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--indices", type=Path, default=None)
    ap.add_argument(
        "--sentences",
        type=Path,
        default=Path("data/corpus/jpn_sentences.tsv.bz2"),
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/labels/tatoeba_indices.jsonl"),
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=80)
    args = ap.parse_args()

    indices_path = args.indices
    if args.download or indices_path is None:
        indices_path = download(args.url, Path("data/corpus/jpn_indices.tar.bz2"))
    if not args.sentences.is_file():
        download(SENTENCES_URL, args.sentences)

    print("Loading sentence map…")
    smap = load_sentence_map(args.sentences)
    print(f"  {len(smap)} jpn sentences")

    kept = 0
    skipped_missing = 0
    skipped_align = 0
    skipped_parse = 0
    skipped_len = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as out:
        for sid, mid, index_text in iter_index_rows(indices_path):
            text = smap.get(sid)
            if not text:
                skipped_missing += 1
                continue
            n = len(text)
            if n < args.min_len or n > args.max_len:
                skipped_len += 1
                continue
            surfaces = extract_surfaces(index_text)
            if not surfaces:
                skipped_parse += 1
                continue
            segs = align_segments(text, surfaces)
            if segs is None:
                skipped_align += 1
                continue
            row = {
                "text": text,
                "segments": segs,
                "teacher": "tatoeba_jpn_indices",
                "tatoeba_id": sid,
                "meaning_id": mid,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            if args.limit > 0 and kept >= args.limit:
                break
            if kept % 20_000 == 0:
                print(f"  kept={kept}", flush=True)

    summary = {
        "kept": kept,
        "skipped_missing": skipped_missing,
        "skipped_align": skipped_align,
        "skipped_parse": skipped_parse,
        "skipped_len": skipped_len,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))
    print("License: Tatoeba / Tanaka indices — CC BY; attribute Tatoeba + authors.")
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
