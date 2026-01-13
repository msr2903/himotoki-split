#!/usr/bin/env python3
"""
Himotoki-Split vs Original Himotoki Comparison Suite

Comprehensive comparison between himotoki-split (lightweight tokenizer)
and original himotoki (full SQLite-based analyzer).

Usage:
    python scripts/compare_himotoki.py                    # Run all tests
    python scripts/compare_himotoki.py --quick            # Quick subset
    python scripts/compare_himotoki.py --sentence "猫が食べる"  # Single sentence
    python scripts/compare_himotoki.py --export results.json    # Export results
"""

import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from enum import Enum

# Local imports
try:
    import test_sentences
except ImportError:
    # If not in path, try adding scripts to path
    sys.path.append(str(Path(__file__).parent))
    import test_sentences

# Add parent to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to import both libraries
try:
    import himotoki_split
    SPLIT_AVAILABLE = True
except ImportError as e:
    print(f"himotoki-split not available: {e}")
    SPLIT_AVAILABLE = False

try:
    # Original himotoki path - check multiple locations
    HIMOTOKI_PATHS = [
        Path.home() / "Projects" / "himotoki",
        PROJECT_ROOT.parent / "himotoki",
    ]
    for path in HIMOTOKI_PATHS:
        if path.exists():
            sys.path.insert(0, str(path))
            break
    import himotoki
    ORIGINAL_AVAILABLE = True
except ImportError as e:
    print(f"original himotoki not available: {e}")
    ORIGINAL_AVAILABLE = False

# =============================================================================
# Data Classes
# =============================================================================

class MatchStatus(Enum):
    MATCH = "match"
    PARTIAL = "partial"
    MISMATCH = "mismatch"
    SPLIT_ERROR = "split_error"
    ORIGINAL_ERROR = "original_error"


@dataclass
class SegmentInfo:
    """Information about a single segment."""
    text: str
    kana: str = ""
    seq: Optional[int] = None
    score: int = 0
    is_compound: bool = False
    pos: str = ""


@dataclass
class SegmentationResult:
    """Result from either tokenizer."""
    segments: List[SegmentInfo]
    total_score: int = 0
    error: Optional[str] = None


@dataclass
class ComparisonResult:
    """Comparison between himotoki-split and original himotoki."""
    sentence: str
    status: MatchStatus
    split_texts: List[str] = field(default_factory=list)
    original_texts: List[str] = field(default_factory=list)
    differences: List[str] = field(default_factory=list)
    time_split: float = 0.0
    time_original: float = 0.0


# Test sentences are now imported from test_sentences.py
TEST_SENTENCES_500 = test_sentences.get_all_sentences()
QUICK_SENTENCES_50 = test_sentences.get_quick_sentences()


# =============================================================================
# Tokenizer Interfaces
# =============================================================================

def run_split(sentence: str) -> SegmentationResult:
    """Run himotoki-split tokenizer."""
    if not SPLIT_AVAILABLE:
        return SegmentationResult(segments=[], error="himotoki-split not available")
    
    try:
        tokens = himotoki_split.tokenize(sentence)
        segments = [
            SegmentInfo(
                text=t.surface,
                kana=t.reading,
                seq=t.base_form_id,
                pos=t.pos,
            )
            for t in tokens
        ]
        return SegmentationResult(segments=segments)
    except Exception as e:
        return SegmentationResult(segments=[], error=str(e))


def run_original(sentence: str) -> SegmentationResult:
    """Run original himotoki analyzer."""
    if not ORIGINAL_AVAILABLE:
        return SegmentationResult(segments=[], error="original himotoki not available")
    
    try:
        results = himotoki.analyze(sentence, limit=1)
        if not results:
            return SegmentationResult(segments=[], error="No segmentation")
        
        words, score = results[0]
        segments = []
        for w in words:
            text = getattr(w, 'text', getattr(w, 'surface', ''))
            kana = getattr(w, 'kana', '')
            if isinstance(kana, list):
                kana = kana[0] if kana else ''
            seq = getattr(w, 'seq', None)
            if isinstance(seq, list):
                seq = seq[0] if seq else None
                
            segments.append(SegmentInfo(
                text=text,
                kana=kana if isinstance(kana, str) else '',
                seq=seq,
                is_compound=getattr(w, 'is_compound', False),
            ))
        
        return SegmentationResult(segments=segments, total_score=score)
    except Exception as e:
        import traceback
        return SegmentationResult(segments=[], error=f"{e}\n{traceback.format_exc()}")


# =============================================================================
# Comparison Logic
# =============================================================================

def compare(sentence: str) -> ComparisonResult:
    """Compare himotoki-split and original himotoki for a sentence."""
    # Run split
    t0 = time.time()
    split_result = run_split(sentence)
    time_split = time.time() - t0
    
    # Run original
    t0 = time.time()
    original_result = run_original(sentence)
    time_original = time.time() - t0
    
    # Extract texts
    split_texts = [s.text for s in split_result.segments]
    original_texts = [s.text for s in original_result.segments]
    
    # Determine status
    differences = []
    if split_result.error:
        status = MatchStatus.SPLIT_ERROR
        differences.append(f"Split error: {split_result.error}")
    elif original_result.error:
        status = MatchStatus.ORIGINAL_ERROR
        differences.append(f"Original error: {original_result.error}")
    elif split_texts == original_texts:
        status = MatchStatus.MATCH
    else:
        status = MatchStatus.MISMATCH
        differences.append(f"original: {original_texts}")
        differences.append(f"split:    {split_texts}")
    
    return ComparisonResult(
        sentence=sentence,
        status=status,
        split_texts=split_texts,
        original_texts=original_texts,
        differences=differences,
        time_split=time_split,
        time_original=time_original,
    )


# =============================================================================
# Output Functions
# =============================================================================

def print_result(result: ComparisonResult):
    """Print a single comparison result."""
    status_icons = {
        MatchStatus.MATCH: "✓",
        MatchStatus.PARTIAL: "~",
        MatchStatus.MISMATCH: "✗",
        MatchStatus.SPLIT_ERROR: "⚠",
        MatchStatus.ORIGINAL_ERROR: "⚠",
    }
    icon = status_icons.get(result.status, "?")
    
    if result.status == MatchStatus.MATCH:
        print(f"  {icon} {result.sentence}: {' | '.join(result.split_texts)}")
    else:
        print(f"  {icon} {result.sentence}")
        for diff in result.differences:
            print(f"      {diff}")


def print_summary(results: List[ComparisonResult]):
    """Print summary statistics."""
    counts = {status: 0 for status in MatchStatus}
    for r in results:
        counts[r.status] += 1
    
    total = len(results)
    match_rate = (counts[MatchStatus.MATCH] + counts[MatchStatus.PARTIAL]) / total * 100 if total else 0
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total:      {total}")
    print(f"  Match:      {counts[MatchStatus.MATCH]}")
    print(f"  Partial:    {counts[MatchStatus.PARTIAL]}")
    print(f"  Mismatch:   {counts[MatchStatus.MISMATCH]}")
    print(f"  Errors:     {counts[MatchStatus.SPLIT_ERROR] + counts[MatchStatus.ORIGINAL_ERROR]}")
    print(f"  Match Rate: {match_rate:.1f}%")
    print("=" * 60)


def export_results(results: List[ComparisonResult], filepath: str):
    """Export results to JSON file."""
    data = []
    for r in results:
        data.append({
            'sentence': r.sentence,
            'status': r.status.value,
            'split_texts': r.split_texts,
            'original_texts': r.original_texts,
            'differences': r.differences,
            'time_split': r.time_split,
            'time_original': r.time_original,
        })
    
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(data)} results to {filepath}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare himotoki-split vs original himotoki")
    parser.add_argument("--quick", action="store_true", help="Run quick test subset")
    parser.add_argument("--sentence", type=str, help="Test a single sentence")
    parser.add_argument("--export", type=str, help="Export results to JSON file")
    args = parser.parse_args()
    
    print("=" * 60)
    print("himotoki-split vs original himotoki Comparison")
    print("=" * 60)
    print()
    
    # Warm up both libraries
    print("Loading himotoki-split...")
    if SPLIT_AVAILABLE:
        himotoki_split.warm_up(verbose=True)
    print()
    
    print("Loading original himotoki...")
    if ORIGINAL_AVAILABLE:
        himotoki.warm_up(verbose=True)
    print()
    
    if not SPLIT_AVAILABLE or not ORIGINAL_AVAILABLE:
        print("Both libraries required for comparison.")
        return 1
    
    # Single sentence test
    if args.sentence:
        result = compare(args.sentence)
        print_result(result)
        return 0 if result.status == MatchStatus.MATCH else 1
    
    # Build sentence list
    if args.quick:
        sentences = QUICK_SENTENCES_50
    else:
        sentences = TEST_SENTENCES_500
    
    # Run comparison
    print(f"Running {len(sentences)} comparisons...")
    print("-" * 60)
    
    results = []
    for sentence in sentences:
        result = compare(sentence)
        results.append(result)
        print_result(result)
    
    print_summary(results)
    
    # Export if requested
    if args.export:
        export_results(results, args.export)
    
    # Return success if all match
    mismatches = sum(1 for r in results if r.status == MatchStatus.MISMATCH)
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
