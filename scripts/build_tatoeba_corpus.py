#!/usr/bin/env python3
"""Build a Japanese sentence corpus from Tatoeba exports.

Downloads ``jpn_sentences.tsv.bz2`` (CC BY 2.0 FR / Tatoeba terms),
filters, dedupes, writes one sentence per line.

Example:
  python scripts/build_tatoeba_corpus.py --download --target 200000 \\
    -o data/corpus/tatoeba_sentences.txt
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import re
import sys
import urllib.request
from pathlib import Path
from typing import Iterator, Optional, Set

DEFAULT_URL = (
    "https://downloads.tatoeba.org/exports/per_language/jpn/jpn_sentences.tsv.bz2"
)
USER_AGENT = "himotoki-split-tatoeba/0.1 (https://github.com/msr2903/himotoki-split)"

_RE_JP = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]")


def _urlopen(url: str, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 100_000:
        print(f"Using existing: {dest} ({dest.stat().st_size // 10**6} MB)")
        return dest
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    with _urlopen(url) as resp, dest.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            print(f"  {dest.stat().st_size // 10**6} MB", flush=True)
    return dest


def iter_tatoeba_sentences(path: Path) -> Iterator[str]:
    """Yield Japanese sentence texts from Tatoeba TSV (.tsv or .tsv.bz2)."""
    opener = bz2.open if str(path).endswith(".bz2") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            # id, lang, text
            lang, text = parts[1], parts[2]
            if lang != "jpn":
                continue
            text = text.strip()
            if text:
                yield text


def jp_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_RE_JP.findall(text)) / len(text)


def is_good(text: str, min_len: int, max_len: int, min_jp: float) -> bool:
    n = len(text)
    if n < min_len or n > max_len:
        return False
    if jp_ratio(text) < min_jp:
        return False
    # Skip obvious meta / URLs
    if "http://" in text or "https://" in text or "www." in text:
        return False
    return True


def build(
    sentences: Iterator[str],
    target: int,
    min_len: int,
    max_len: int,
    min_jp: float,
) -> list[str]:
    seen: Set[str] = set()
    out: list[str] = []
    scanned = 0
    for text in sentences:
        scanned += 1
        if not is_good(text, min_len, max_len, min_jp):
            continue
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if target > 0 and len(out) >= target:
            break
        if scanned % 50_000 == 0:
            print(f"  scanned={scanned} kept={len(out)}", flush=True)
    print(f"Done: scanned={scanned} kept={len(out)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument(
        "--tsv",
        type=Path,
        help="Local jpn_sentences.tsv or .tsv.bz2",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/corpus/tatoeba_sentences.txt"),
    )
    ap.add_argument("--target", type=int, default=200_000)
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--min-jp", type=float, default=0.55)
    args = ap.parse_args()

    tsv = args.tsv
    if args.download or tsv is None:
        dest = Path("data/corpus/jpn_sentences.tsv.bz2")
        tsv = download(args.url, dest)
    if not tsv.is_file():
        print(f"Missing Tatoeba TSV: {tsv}", file=sys.stderr)
        return 1

    sents = build(
        iter_tatoeba_sentences(tsv),
        target=args.target,
        min_len=args.min_len,
        max_len=args.max_len,
        min_jp=args.min_jp,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sents) + ("\n" if sents else ""), encoding="utf-8")
    print(f"Wrote {len(sents)} sentences -> {args.output}")
    print("License: Tatoeba sentences are CC BY 2.0 FR — attribute Tatoeba / authors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
