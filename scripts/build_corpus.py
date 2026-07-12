#!/usr/bin/env python3
"""Build a cleaned Japanese sentence corpus from a Wikipedia dump.

Streams jawiki pages-articles XML (.bz2 or .xml), strips markup lightly,
sentence-splits, filters, dedupes, and writes one sentence per line until
``--target`` is reached.

Example:
  python scripts/build_corpus.py --target 100000 -o data/corpus/sentences.txt
  python scripts/build_corpus.py --dump /path/to/jawiki-latest-pages-articles.xml.bz2
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import re
import sys
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Iterator, Optional, Set

DEFAULT_DUMP_URL = (
    "https://dumps.wikimedia.org/jawiki/latest/"
    "jawiki-latest-pages-articles.xml.bz2"
)
USER_AGENT = "himotoki-split-corpus/0.1 (https://github.com/msr2903/himotoki-split)"


def _urlopen(url: str, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)

# Conservative cleanup
_RE_NOWIKI = re.compile(r"<nowiki>.*?</nowiki>", re.I | re.S)
_RE_REF = re.compile(r"<ref\b[^>]*>.*?</ref>", re.I | re.S)
_RE_REF_EMPTY = re.compile(r"<ref\b[^/]*/>", re.I)
_RE_HTML = re.compile(r"<[^>]+>")
_RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_RE_LINK_LABELED = re.compile(r"\[\[[^\]]*\|([^\]]+)\]\]")
_RE_LINK = re.compile(r"\[\[([^\]]+)\]\]")
_RE_EXTERNAL = re.compile(r"\[https?://[^\]\s]+(?:\s+[^\]]+)?\]")
_RE_URL = re.compile(r"https?://\S+")
_RE_FILE = re.compile(r"\[\[(?:File|Image|ファイル|画像):[^\]]*\]\]", re.I)
_RE_CATEGORY = re.compile(r"\[\[Category:[^\]]*\]\]", re.I)
_RE_HEADING = re.compile(r"^=+\s*.*?\s*=+$", re.M)
_RE_TABLE_CELL = re.compile(r"^\|.*$", re.M)
_RE_BOLD_ITAL = re.compile(r"'{2,}")
_RE_WS = re.compile(r"\s+")
_RE_SENT_SPLIT = re.compile(r"(?<=[。！？])|(?<=\n)")

_RE_JP = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def _open_dump(path: Path):
    if str(path).endswith(".bz2"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def download_dump(url: str, dest: Path, progress: bool = True) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        if progress:
            print(f"Using existing dump: {dest} ({dest.stat().st_size // 10**6} MB)")
        return dest

    if progress:
        print(f"Downloading {url}")
        print(f"  -> {dest}")

    with _urlopen(url) as resp, dest.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress and total:
                pct = min(100.0, 100.0 * done / total)
                print(
                    f"\r  {pct:5.1f}% ({done // 10**6} / {total // 10**6} MB)",
                    end="",
                    flush=True,
                )
    if progress:
        print()
    return dest


def iter_wiki_pages_streaming(url: str) -> Iterator[str]:
    """Stream a remote .bz2 dump and yield page wikitext (no full download)."""
    print(f"Streaming dump from {url}")
    resp = _urlopen(url)
    with bz2.open(resp, "rt", encoding="utf-8", errors="replace") as f:
        yield from iter_wiki_pages_from_file(f)


def strip_wikitext(text: str) -> str:
    text = _RE_NOWIKI.sub(" ", text)
    text = _RE_REF.sub(" ", text)
    text = _RE_REF_EMPTY.sub(" ", text)
    text = _RE_FILE.sub(" ", text)
    text = _RE_CATEGORY.sub(" ", text)
    # Nested templates: run a few passes
    for _ in range(4):
        nxt = _RE_TEMPLATE.sub(" ", text)
        if nxt == text:
            break
        text = nxt
    text = _RE_LINK_LABELED.sub(r"\1", text)
    text = _RE_LINK.sub(r"\1", text)
    text = _RE_EXTERNAL.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_HTML.sub(" ", text)
    text = _RE_HEADING.sub(" ", text)
    text = _RE_TABLE_CELL.sub(" ", text)
    text = _RE_BOLD_ITAL.sub("", text)
    text = text.replace("{", " ").replace("}", " ")
    text = _RE_WS.sub(" ", text)
    return text.strip()


def japanese_ratio(text: str) -> float:
    if not text:
        return 0.0
    jp = len(_RE_JP.findall(text))
    return jp / len(text)


def normalize_key(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def is_good_sentence(text: str, min_len: int, max_len: int, min_jp: float) -> bool:
    if not (min_len <= len(text) <= max_len):
        return False
    if japanese_ratio(text) < min_jp:
        return False
    if "http" in text or "www." in text:
        return False
    if text.count("|") > 2 or text.count("=") > 2:
        return False
    # Avoid list-y / markup leftovers
    if text.startswith("*") or text.startswith("#") or text.startswith(":"):
        return False
    return True


def split_sentences(text: str) -> Iterator[str]:
    parts = _RE_SENT_SPLIT.split(text)
    buf = ""
    for part in parts:
        if part is None:
            continue
        buf += part
        if part.endswith(("。", "！", "？")) or "\n" in part:
            s = _RE_WS.sub(" ", buf).strip()
            buf = ""
            if s:
                yield s
    s = _RE_WS.sub(" ", buf).strip()
    if s:
        yield s


def iter_wiki_pages_from_file(f) -> Iterator[str]:
    """Yield raw wikitext for main-namespace pages from an open text stream."""
    context = ET.iterparse(f, events=("end",))
    for _event, elem in context:
        tag = elem.tag
        if tag.endswith("page"):
            ns = "0"
            title = ""
            text = ""
            for child in elem:
                ctag = child.tag
                if ctag.endswith("ns") and child.text:
                    ns = child.text
                elif ctag.endswith("title") and child.text:
                    title = child.text
                elif ctag.endswith("revision"):
                    for rev_child in child:
                        if rev_child.tag.endswith("text") and rev_child.text:
                            text = rev_child.text
            elem.clear()
            if ns != "0" or not text:
                continue
            if title.startswith(
                (
                    "Wikipedia:",
                    "Template:",
                    "File:",
                    "Help:",
                    "Portal:",
                    "モジュール:",
                    "テンプレート:",
                )
            ):
                continue
            yield text
        elif tag.endswith(("mediawiki",)):
            elem.clear()


def iter_wiki_pages(dump_path: Path) -> Iterator[str]:
    """Yield raw wikitext for main-namespace pages."""
    with _open_dump(dump_path) as f:
        yield from iter_wiki_pages_from_file(f)


def build_sentences_from_pages(
    pages: Iterator[str],
    target: int,
    min_len: int = 8,
    max_len: int = 80,
    min_jp: float = 0.5,
) -> list[str]:
    seen: Set[str] = set()
    out: list[str] = []
    page_count = 0
    for wikitext in pages:
        page_count += 1
        plain = strip_wikitext(wikitext)
        if not plain:
            continue
        for sent in split_sentences(plain):
            sent = normalize_key(sent)
            if not is_good_sentence(sent, min_len, max_len, min_jp):
                continue
            key = hashlib.sha1(sent.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append(sent)
            if len(out) >= target:
                print(f"Reached target {target} after {page_count} pages")
                return out
        if page_count % 500 == 0:
            print(f"  pages={page_count} sentences={len(out)}", flush=True)
    print(f"Source exhausted: pages={page_count} sentences={len(out)}")
    return out


def build_sentences(
    dump_path: Path,
    target: int,
    min_len: int = 8,
    max_len: int = 80,
    min_jp: float = 0.5,
) -> list[str]:
    return build_sentences_from_pages(
        iter_wiki_pages(dump_path),
        target=target,
        min_len=min_len,
        max_len=max_len,
        min_jp=min_jp,
    )

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dump",
        type=Path,
        help="Path to jawiki pages-articles.xml or .xml.bz2",
    )
    ap.add_argument(
        "--download",
        action="store_true",
        help=f"Download dump from Wikimedia (default URL) into data/corpus/",
    )
    ap.add_argument(
        "--url",
        default=DEFAULT_DUMP_URL,
        help="Dump URL when --download is set",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/corpus/sentences.txt"),
        help="Output sentence file (one per line)",
    )
    ap.add_argument("--target", type=int, default=100_000)
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--min-jp-ratio", type=float, default=0.5)
    ap.add_argument(
        "--stream",
        action="store_true",
        help="Stream remote dump until --target (no full local dump file)",
    )
    args = ap.parse_args()

    use_stream = bool(args.stream or (args.download and args.dump is None))
    if use_stream:
        url = args.url
        print(f"Extracting up to {args.target} sentences (stream mode)")
        try:
            sentences = build_sentences_from_pages(
                iter_wiki_pages_streaming(url),
                target=args.target,
                min_len=args.min_len,
                max_len=args.max_len,
                min_jp=args.min_jp_ratio,
            )
        except Exception as exc:
            print(f"Stream failed: {exc}", file=sys.stderr)
            return 1
    else:
        dump = args.dump
        if dump is None:
            print("Provide --dump PATH, or --stream / --download", file=sys.stderr)
            return 1
        if not dump.is_file():
            print(f"Dump not found: {dump}", file=sys.stderr)
            return 1

        print(f"Extracting up to {args.target} sentences from {dump}")
        sentences = build_sentences(
            dump,
            target=args.target,
            min_len=args.min_len,
            max_len=args.max_len,
            min_jp=args.min_jp_ratio,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(sentences) + ("\n" if sentences else ""), encoding="utf-8"
    )
    print(f"Wrote {len(sentences)} sentences to {args.output}")
    print(
        "Note: Wikipedia text is CC BY-SA; derived silver labels inherit usage constraints."
    )
    return 0 if sentences else 1

if __name__ == "__main__":
    raise SystemExit(main())
