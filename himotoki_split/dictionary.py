"""
Binary Dictionary for himotoki-split.

This module provides a memory-mapped binary dictionary for fast word lookup.
The dictionary is built from JMdict XML and contains:
- Surface forms (keys)
- Sequence IDs, costs, POS, and base form information (values)

The dictionary is stored as a marisa_trie.RecordTrie for efficient prefix
matching and memory-mapped access.
"""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import marisa_trie

# ============================================================================
# Binary Record Schema
# ============================================================================
# Each entry stores:
#   - seq: int32 (4 bytes) - JMdict sequence ID
#   - cost: int16 (2 bytes) - Pre-calculated score/cost for Viterbi
#   - pos_id: uint8 (1 byte) - Part of speech ID
#   - conj_type: uint8 (1 byte) - Conjugation type (0 = root form)
#   - base_seq: int32 (4 bytes) - Base form sequence ID (0 if root)
#
# Total: 12 bytes per entry
# Format string: little-endian int32, int16, uint8, uint8, int32

RECORD_FORMAT = "<ihBBI"
RECORD_SIZE = struct.calcsize(RECORD_FORMAT)  # Should be 12


@dataclass(slots=True)
class WordEntry:
    """
    A word entry from the binary dictionary.
    
    Attributes:
        surface: The surface form (text as it appears)
        seq: JMdict sequence ID
        cost: Pre-calculated cost for scoring
        pos_id: Part of speech ID (see POS_ID_MAP)
        conj_type: Conjugation type (0 = dictionary form)
        base_seq: Base form sequence ID (same as seq if dictionary form)
    """
    surface: str
    seq: int
    cost: int
    pos_id: int
    conj_type: int
    base_seq: int
    
    @property
    def is_root(self) -> bool:
        """True if this is a dictionary form (not conjugated)."""
        return self.base_seq == 0 or self.base_seq == self.seq
    
    @property
    def base_form_id(self) -> int:
        """Get the base form sequence ID."""
        return self.base_seq if self.base_seq != 0 else self.seq


# ============================================================================
# POS ID Mapping
# ============================================================================
# Maps POS strings to compact IDs for storage

POS_ID_MAP = {
    'n': 1, 'n-adv': 2, 'n-pref': 3, 'n-suf': 4, 'n-t': 5,
    'v1': 10, 'v1-s': 11, 'v5aru': 12, 'v5b': 13, 'v5g': 14,
    'v5k': 15, 'v5k-s': 16, 'v5m': 17, 'v5n': 18, 'v5r': 19,
    'v5r-i': 20, 'v5s': 21, 'v5t': 22, 'v5u': 23, 'v5u-s': 24,
    'v5uru': 25, 'vk': 26, 'vs': 27, 'vs-i': 28, 'vs-s': 29, 'vz': 30,
    'adj-i': 40, 'adj-ix': 41, 'adj-na': 42, 'adj-no': 43,
    'adj-pn': 44, 'adj-t': 45, 'adj-f': 46,
    'adv': 50, 'adv-to': 51,
    'aux': 60, 'aux-v': 61, 'aux-adj': 62,
    'conj': 70, 'cop': 71, 'ctr': 72, 'exp': 73, 'int': 74,
    'pn': 80, 'pref': 81, 'prt': 82, 'suf': 83, 'unc': 84,
}

# Reverse mapping for lookup
ID_TO_POS = {v: k for k, v in POS_ID_MAP.items()}


def get_pos_id(pos: str) -> int:
    """Get compact ID for a POS string."""
    return POS_ID_MAP.get(pos, 0)


def get_pos_name(pos_id: int) -> str:
    """Get POS string from compact ID."""
    return ID_TO_POS.get(pos_id, 'unk')


# ============================================================================
# Dictionary Loading
# ============================================================================

# Module-level singleton
_DICTIONARY: Optional[marisa_trie.RecordTrie] = None
_BASE_FORMS: Optional[dict] = None  # seq -> base_form text
_KANA_READINGS: Optional[dict] = None  # seq -> kana reading


def get_dictionary_path() -> Path:
    """Get the default dictionary path."""
    return Path(__file__).parent / "data" / "himotoki.dic"


def get_base_forms_path() -> Path:
    """Get the path to base forms mapping."""
    return Path(__file__).parent / "data" / "base_forms.bin"


def get_kana_readings_path() -> Path:
    """Get the path to kana readings mapping."""
    return Path(__file__).parent / "data" / "kana_readings.bin"


def is_dictionary_loaded() -> bool:
    """Check if dictionary is loaded."""
    return _DICTIONARY is not None


def load_dictionary(path: Optional[Path] = None) -> marisa_trie.RecordTrie:
    """
    Load the binary dictionary.
    
    The dictionary is memory-mapped for instant loading and low memory usage.
    
    Args:
        path: Path to the .dic file. Uses default if not specified.
        
    Returns:
        The loaded RecordTrie
        
    Raises:
        FileNotFoundError: If dictionary file doesn't exist
    """
    global _DICTIONARY
    
    if _DICTIONARY is not None:
        return _DICTIONARY
    
    if path is None:
        path = get_dictionary_path()
    
    if not path.exists():
        raise FileNotFoundError(
            f"Dictionary not found at {path}. "
            "Run 'python -m himotoki_split.build' to build it."
        )
    
    _DICTIONARY = marisa_trie.RecordTrie(RECORD_FORMAT)
    _DICTIONARY.mmap(str(path))
    
    return _DICTIONARY


def lookup(surface: str) -> List[WordEntry]:
    """
    Look up a surface form in the dictionary.
    
    Args:
        surface: The text to look up
        
    Returns:
        List of matching WordEntry objects
    """
    global _DICTIONARY
    
    if _DICTIONARY is None:
        load_dictionary()
    
    results = []
    
    try:
        records = _DICTIONARY.get(surface, [])
        for record in records:
            seq, cost, pos_id, conj_type, base_seq = record
            results.append(WordEntry(
                surface=surface,
                seq=seq,
                cost=cost,
                pos_id=pos_id,
                conj_type=conj_type,
                base_seq=base_seq,
            ))
    except KeyError:
        pass
    
    return results


def lookup_prefix(prefix: str) -> List[Tuple[str, WordEntry]]:
    """
    Look up all words starting with a prefix.
    
    Args:
        prefix: The prefix to search for
        
    Returns:
        List of (surface, WordEntry) tuples
    """
    global _DICTIONARY
    
    if _DICTIONARY is None:
        load_dictionary()
    
    results = []
    
    for surface, records in _DICTIONARY.items(prefix):
        for record in records:
            seq, cost, pos_id, conj_type, base_seq = record
            results.append((surface, WordEntry(
                surface=surface,
                seq=seq,
                cost=cost,
                pos_id=pos_id,
                conj_type=conj_type,
                base_seq=base_seq,
            )))
    
    return results


def contains(surface: str) -> bool:
    """Check if a surface form exists in the dictionary."""
    global _DICTIONARY
    
    if _DICTIONARY is None:
        load_dictionary()
    
    return surface in _DICTIONARY


def has_prefix(prefix: str) -> bool:
    """Check if any word starts with the given prefix."""
    global _DICTIONARY
    
    if _DICTIONARY is None:
        load_dictionary()
    
    try:
        next(iter(_DICTIONARY.iterkeys(prefix)))
        return True
    except StopIteration:
        return False


def get_dictionary_size() -> int:
    """Get the number of entries in the dictionary."""
    global _DICTIONARY
    
    if _DICTIONARY is None:
        return 0
    
    return len(_DICTIONARY)


def load_kana_readings() -> dict:
    """Load kana readings from binary file."""
    global _KANA_READINGS
    
    if _KANA_READINGS is not None:
        return _KANA_READINGS
    
    path = get_kana_readings_path()
    if not path.exists():
        _KANA_READINGS = {}
        return _KANA_READINGS
    
    _KANA_READINGS = {}
    with open(path, 'rb') as f:
        count = struct.unpack('<I', f.read(4))[0]
        for _ in range(count):
            seq = struct.unpack('<I', f.read(4))[0]
            text_len = struct.unpack('<H', f.read(2))[0]
            text = f.read(text_len).decode('utf-8')
            _KANA_READINGS[seq] = text
    
    return _KANA_READINGS


def get_kana_reading(seq: int) -> Optional[str]:
    """Get kana reading for a seq number."""
    global _KANA_READINGS
    
    if _KANA_READINGS is None:
        load_kana_readings()
    
    return _KANA_READINGS.get(seq)


def unload_dictionary():
    """Unload the dictionary to free memory."""
    global _DICTIONARY, _BASE_FORMS, _KANA_READINGS
    _DICTIONARY = None
    _BASE_FORMS = None
    _KANA_READINGS = None
