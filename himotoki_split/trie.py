"""
Word trie for fast dictionary surface form lookup.

Uses the binary dictionary (himotoki.dic) for memory-efficient storage.
This module provides prefix trie operations via the RecordTrie.
"""

from typing import Optional

from himotoki_split.dictionary import (
    load_dictionary,
    is_dictionary_loaded,
    contains,
    has_prefix,
    get_dictionary_size,
)


def get_word_trie():
    """Get the initialized trie. Loads if necessary."""
    return load_dictionary()


def is_trie_ready() -> bool:
    """Check if trie has been initialized."""
    return is_dictionary_loaded()


def init_word_trie(session=None):
    """
    Initialize the word trie.
    
    Note: session parameter kept for API compatibility but is ignored.
    Uses binary dictionary instead.
        
    Returns:
        The RecordTrie
    """
    return load_dictionary()


def trie_contains(word: str) -> bool:
    """
    Check if word exists in the trie.
    
    Args:
        word: Surface form to check
        
    Returns:
        True if word exists in dictionary, False otherwise
    """
    return contains(word)


def trie_has_prefix(prefix: str) -> bool:
    """
    Check if any word in the trie starts with the given prefix.
    
    Args:
        prefix: Prefix to check
        
    Returns:
        True if any word starts with prefix, False otherwise
    """
    return has_prefix(prefix)


def get_trie_size() -> int:
    """Get number of entries in the trie."""
    return get_dictionary_size()
