"""
Lightweight data structures for dictionary lookups.

These dataclasses replace the heavy ORM objects (KanaText, KanjiText) and
are populated from the binary dictionary.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any, NamedTuple


class RawKanaReading(NamedTuple):
    """
    Lightweight replacement for KanaText ORM object.
    Used for compatibility with existing code.
    """
    id: int
    seq: int
    text: str
    ord: int
    common: Optional[int]
    best_kanji: Optional[str]


class RawKanjiReading(NamedTuple):
    """
    Lightweight replacement for KanjiText ORM object.
    Used for compatibility with existing code.
    """
    id: int
    seq: int
    text: str
    ord: int
    common: Optional[int]
    best_kana: Optional[str]


@dataclass(slots=True)
class DictWord:
    """
    A word from the binary dictionary.
    
    This is the primary data structure for himotoki-split lookups.
    """
    surface: str
    seq: int
    cost: int
    pos_id: int
    conj_type: int
    base_seq: int
    
    @property
    def text(self) -> str:
        """Alias for surface (compatibility)."""
        return self.surface
    
    @property
    def common(self) -> Optional[int]:
        """Cost as common score (lower = more common)."""
        return self.cost
    
    @property
    def ord(self) -> int:
        """Order (derived from cost)."""
        return 0
    
    @property
    def is_root(self) -> bool:
        """True if this is a dictionary form."""
        return self.conj_type == 0 or self.base_seq == self.seq
    
    @property
    def base_form_id(self) -> int:
        """Get base form sequence ID."""
        return self.base_seq if self.base_seq != 0 else self.seq


@dataclass
class ConjProp:
    """Conjugation property (minimal version)."""
    conj_type: int
    pos: str = ""
    neg: Optional[bool] = None
    fml: Optional[bool] = None


@dataclass 
class ConjData:
    """Conjugation data for a word match."""
    seq: int
    from_seq: int
    via: Optional[int] = None
    prop: Optional[ConjProp] = None
    src_map: List[Tuple[str, str]] = field(default_factory=list)
