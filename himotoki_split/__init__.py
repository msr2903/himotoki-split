"""
himotoki-split: Lightweight Japanese Morphological Analyzer

A pure tokenizer like Sudachipy. No database required.
Uses a compact binary dictionary (~27MB) for instant startup.

Basic Usage:
    import himotoki_split
    
    # Tokenize text
    tokens = himotoki_split.tokenize("今日は天気がいいです")
    for token in tokens:
        print(f"{token.surface} -> {token.base_form} ({token.pos})")
"""

import time
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

__version__ = "0.1.0"


# =============================================================================
# Token Data Structure
# =============================================================================

@dataclass(slots=True)
class Token:
    """
    A token (word) from morphological analysis.
    
    Attributes:
        surface: The surface form (as it appears in text)
        reading: The reading in hiragana
        pos: Part of speech (e.g., "n", "v1", "adj-i")
        base_form: The dictionary form (e.g., "食べる" for "食べた")
        base_form_id: JMdict sequence ID for dictionary lookup
        start: Start position in the original text
        end: End position in the original text
    """
    surface: str
    reading: str
    pos: str
    base_form: str
    base_form_id: int
    start: int
    end: int
    
    def __repr__(self) -> str:
        return f"Token({self.surface!r}, base={self.base_form!r}, pos={self.pos!r})"


# =============================================================================
# Main API
# =============================================================================

def tokenize(text: str) -> List[Token]:
    """
    Tokenize Japanese text into morphemes.
    
    This is the main entry point for text analysis.
    
    Args:
        text: Japanese text to tokenize (must be non-empty)
        
    Returns:
        List of Token objects
        
    Raises:
        ValueError: If text is empty or whitespace-only
        
    Example:
        >>> import himotoki_split
        >>> tokens = himotoki_split.tokenize("食べました")
        >>> for t in tokens:
        ...     print(f"{t.surface} -> {t.base_form}")
        食べ -> 食べる
        ました -> ます
    """
    if not text or not text.strip():
        raise ValueError("text must be non-empty and not whitespace-only")
    
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)
    
    from himotoki_split.tokenizer import tokenize_text
    return tokenize_text(text)


def analyze(text: str, limit: int = 1) -> List[Tuple[List[Token], float]]:
    """
    Analyze Japanese text and return multiple segmentation candidates.
    
    Args:
        text: Japanese text to analyze
        limit: Maximum number of results to return
        
    Returns:
        List of (tokens, score) tuples, sorted by score descending
        
    Example:
        >>> results = himotoki_split.analyze("今日は", limit=3)
        >>> for tokens, score in results:
        ...     print(f"Score {score}: {[t.surface for t in tokens]}")
    """
    if not text or not text.strip():
        raise ValueError("text must be non-empty and not whitespace-only")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    
    text = unicodedata.normalize('NFC', text)
    
    from himotoki_split.tokenizer import analyze_text
    return analyze_text(text, limit=limit)


def warm_up(verbose: bool = False) -> Tuple[float, dict]:
    """
    Pre-load the dictionary for optimal performance.
    
    The dictionary is memory-mapped, so this is fast (~10ms).
    
    Args:
        verbose: If True, print timing information
        
    Returns:
        Tuple of (total_time_seconds, timing_details_dict)
    """
    from himotoki_split.dictionary import load_dictionary, get_dictionary_size
    
    timings = {}
    total_start = time.perf_counter()
    
    if verbose:
        print("Loading himotoki-split dictionary...")
    
    t0 = time.perf_counter()
    load_dictionary()
    timings['dictionary'] = (time.perf_counter() - t0) * 1000
    
    if verbose:
        print(f"  Dictionary:     {timings['dictionary']:>7.1f}ms ({get_dictionary_size():,} entries)")
    
    total_time = time.perf_counter() - total_start
    timings['total'] = total_time * 1000
    
    if verbose:
        print(f"Total warm-up:    {timings['total']:>7.1f}ms")
    
    return total_time, timings


def get_version() -> str:
    """Get the library version."""
    return __version__


def get_conjugation_hint(text: str) -> Optional[str]:
    """
    Get a learner-friendly explanation for a grammar pattern.
    
    Args:
        text: Japanese grammar pattern text
        
    Returns:
        Human-readable explanation if found, None otherwise
        
    Example:
        >>> himotoki_split.get_conjugation_hint("なければならない")
        "must; have to"
        >>> himotoki_split.get_conjugation_hint("てもいい")
        "may; it's okay to"
    """
    from himotoki_split.conjugation_hints import get_conjugation_hint as _get_hint
    return _get_hint(text)


def romanize(text: str) -> str:
    """
    Convert kana text to romaji.
    
    Args:
        text: Kana text to romanize
        
    Returns:
        Romanized text
        
    Example:
        >>> himotoki_split.romanize("こんにちは")
        "konnichiha"
    """
    from himotoki_split.characters import romanize_word
    return romanize_word(text)


# =============================================================================
# Async API
# =============================================================================

# Thread pool for async operations
_executor = None
_executor_lock = None

def _get_executor():
    """Get or create the thread pool executor."""
    global _executor, _executor_lock
    import threading
    from concurrent.futures import ThreadPoolExecutor
    
    if _executor_lock is None:
        _executor_lock = threading.Lock()
    
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="himotoki")
    
    return _executor


class AnalysisTimeoutError(Exception):
    """Raised when async analysis times out."""
    pass


class TextTooLongError(Exception):
    """Raised when input text exceeds maximum length."""
    pass


async def tokenize_async(
    text: str,
    timeout: float = 30.0,
) -> List[Token]:
    """
    Tokenize Japanese text asynchronously.
    
    Args:
        text: Japanese text to tokenize
        timeout: Maximum time in seconds (default 30s)
        
    Returns:
        List of Token objects
        
    Raises:
        AnalysisTimeoutError: If tokenization exceeds timeout
        ValueError: If text is empty
        
    Example:
        >>> import asyncio
        >>> tokens = asyncio.run(himotoki_split.tokenize_async("今日は"))
    """
    import asyncio
    
    loop = asyncio.get_event_loop()
    executor = _get_executor()
    
    try:
        future = loop.run_in_executor(executor, tokenize, text)
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        raise AnalysisTimeoutError(f"Tokenization timed out after {timeout}s")


async def analyze_async(
    text: str,
    limit: int = 1,
    timeout: float = 30.0,
) -> List[Tuple[List[Token], float]]:
    """
    Analyze Japanese text asynchronously.
    
    Args:
        text: Japanese text to analyze
        limit: Maximum number of results
        timeout: Maximum time in seconds (default 30s)
        
    Returns:
        List of (tokens, score) tuples
        
    Raises:
        AnalysisTimeoutError: If analysis exceeds timeout
        
    Example:
        >>> import asyncio
        >>> results = asyncio.run(himotoki_split.analyze_async("今日は", limit=3))
    """
    import asyncio
    
    loop = asyncio.get_event_loop()
    executor = _get_executor()
    
    try:
        future = loop.run_in_executor(executor, lambda: analyze(text, limit))
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        raise AnalysisTimeoutError(f"Analysis timed out after {timeout}s")


def shutdown():
    """
    Shutdown the thread pool executor.
    
    Call this when your application is shutting down to cleanly
    release resources.
    """
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None


# =============================================================================
# Session Context (for batch processing)
# =============================================================================

from contextlib import contextmanager

@contextmanager
def session_context():
    """
    Context manager for batch tokenization.
    
    Pre-loads the dictionary and provides optimized tokenization
    for processing multiple texts.
    
    Example:
        >>> with himotoki_split.session_context():
        ...     for text in texts:
        ...         tokens = himotoki_split.tokenize(text)
        ...         # process tokens
    """
    # Pre-load dictionary
    from himotoki_split.dictionary import load_dictionary
    load_dictionary()
    
    try:
        yield
    finally:
        # No cleanup needed for memory-mapped dictionary
        pass


# =============================================================================
# Module-level exports
# =============================================================================

__all__ = [
    # Data classes
    "Token",
    # Sync API
    "tokenize", 
    "analyze",
    "warm_up",
    "get_version",
    "get_conjugation_hint",
    "romanize",
    # Async API
    "tokenize_async",
    "analyze_async",
    "shutdown",
    # Batch processing
    "session_context",
    # Exceptions
    "AnalysisTimeoutError",
    "TextTooLongError",
    # Version
    "__version__",
]