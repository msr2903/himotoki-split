"""
Split definitions module for himotoki-split.
Ported from original himotoki's splits.py (dict-split.lisp).

Splits allow compound words to be scored as the sum of their parts.
For example: 一人で → 一人 + で

This version is adapted for the binary dictionary approach,
without requiring SQLite database lookups.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple, Any, Callable

from himotoki_split.dictionary import lookup, contains, WordEntry


# ============================================================================
# Split Result Types
# ============================================================================

@dataclass
class SplitPart:
    """A part of a split compound word."""
    text: str
    entry: Optional[WordEntry] = None
    
    def __repr__(self):
        return f"<SplitPart({self.text})>"


@dataclass
class SplitResult:
    """
    Result of splitting a word.
    
    Supports score modification via score_bonus.
    """
    parts: List[SplitPart]
    score_bonus: int = 0
    
    def __repr__(self):
        texts = [p.text for p in self.parts]
        return f"<SplitResult({' + '.join(texts)}, bonus={self.score_bonus})>"


# ============================================================================
# Seq-based Split Maps
# ============================================================================

# Map: compound seq -> (main_seq, main_length, suffix_text, suffix_seq, score)
# This pre-defines how to split specific JMdict entries

# -de splits: compound_seq -> (main_seq, suffix_seq='で', score)
DE_SPLITS: Dict[int, Tuple[int, int]] = {
    1163700: (1576150, 20),    # 一人で -> 一人 + で
    1611020: (1577100, 20),    # 何で -> 何 + で
    1004800: (1628530, 20),    # これで -> これ + で
    2810720: (1004820, 20),    # 此れまでで
    1006840: (1006880, 20),    # その上で
    1530610: (1530600, 20),    # 無断で
    1245390: (1245290, 20),    # 空で
    2719270: (1445430, 20),    # 土足で
    1189420: (2416780, 20),    # 何用で
    1272220: (1592990, 20),    # 交代で
    1311360: (1311350, 20),    # 私費で
    1368500: (1368490, 20),    # 人前で
    1395670: (1395660, 20),    # 全体で
    1417790: (1417780, 20),    # 単独で
    1454270: (1454260, 20),    # 道理で
    1479100: (1679020, 20),    # 半眼で
    1510140: (1680900, 20),    # 別封で
    1518550: (1529560, 20),    # 無しで
    1531420: (1531410, 20),    # 名義で
    1597400: (1585205, 20),    # 力尽くで
    1679990: (2582460, 20),    # 抜き足で
    1682060: (2085340, 20),    # 金ずくで
    1736650: (1611710, 20),    # 水入らずで
    1865020: (1590150, 20),    # 陰で
    1878880: (2423450, 20),    # 差しで
    2126220: (1802920, 20),    # 捩じり鉢巻きで
    2136520: (2005870, 20),    # もう少しで
    2513590: (2513650, 20),    # 詰め開きで
    2771850: (2563780, 20),    # 気にしないで
    2810800: (1587590, 20),    # 今までで
    1343110: (1343100, 20),    # ところで
    1270210: (1001640, 20),    # お陰で
}

# -通り splits: compound_seq -> (main_seq, toori_seq, score)
TOORI_SPLITS: Dict[int, Tuple[int, int, int]] = {
    1260990: (1260670, 1432930, 50),    # 元通り
    1414570: (2082450, 1432930, 50),    # 大通り
    1424950: (1620400, 1432930, 50),    # 中通り (ちゅう通り)
    1424960: (1423310, 1432930, 50),    # 中通り (なか通り)
    1820790: (1250090, 1432930, 50),    # 型通り
    1489800: (1489340, 1432930, 50),    # 表通り
    1523010: (1522150, 1432930, 50),    # 本通り
    1808080: (1604890, 1432930, 50),    # 目通り
    1368820: (1580640, 1432930, 50),    # 人通り
    1550490: (1550190, 1432930, 50),    # 裏通り
    1619440: (2069220, 1432930, 50),    # 素通り
    1164910: (2821500, 1432920, 50),    # 一通り
    1462720: (1461140, 1432920, 50),    # 二通り
}

# ど- prefix splits: compound_seq -> (rest_seq, score)
DO_SPLITS: Dict[int, Tuple[int, int]] = {
    2142710: (1185200, 30),    # ど下手
    2803190: (1595630, 30),    # どすけべ
    2142680: (1290210, 30),    # ど根性
    2523480: (1442750, 30),    # ど田舎
}

# し- (する conjugation) splits: compound_seq -> (rest_seq, score)
SHI_SPLITS: Dict[int, Tuple[int, int]] = {
    1005700: (1156990, 30),    # し易い
    1005830: (1370760, 30),    # し吹く
    1157200: (2772730, 30),    # し難い
    1157220: (1195970, 30),    # し過ぎる
    1157230: (1284430, 30),    # し合う
    1157280: (1370090, 30),    # し尽す
    1157310: (1405800, 30),    # し続ける
    1304890: (1256520, 30),    # し兼ねる
    1304960: (1307550, 30),    # し始める
    1305110: (1338180, 30),    # し出す
    1305280: (1599390, 30),    # し直す
    1305290: (1212670, 30),    # し慣れる
    1594300: (1596510, 30),    # し損なう
    1594310: (1406680, 30),    # し損じる
    1594460: (1372620, 30),    # し遂げる
    1594580: (1277100, 30),    # し向ける
    2518250: (1332760, 30),    # し終える
    1157240: (1600260, 30),    # し残す
    1304820: (1207610, 30),    # し掛ける
    2858937: (1406690, 30),    # し損ねる
}

# Complex splits: compound_seq -> (parts_seqs, score)
COMPLEX_SPLITS: Dict[int, Tuple[List[int], int]] = {
    # なくなる: 無く + なる
    1529550: ([1529520, 1375610], 30),
    # という: と + 言う
    1922760: ([1008490, 1587040], 20),
    # じゃない: じゃ + ない
    2755350: ([2089020, 1529520], 10),
    # なら: だ conditional
    1009470: ([2089020], 1),
    # 気がつく: 気 + が + つく
    1591050: ([1221520, 2028930, 1495740], 100),
    # 気のせい: 気 + の + せい
    1221750: ([1221520, 1469800, 1610040], 100),
}

# Segment splits (for path expansion): compound_seq -> (parts_seqs, score)
# These have negative scores to discourage over-splitting
SEGMENT_SPLITS: Dict[int, Tuple[List[int], int]] = {
    # ところが: ところ + が
    1008570: ([1343100, 2028930], -10),
    # ところで: ところ + で
    # Note: 1343110 is also in DE_SPLITS, segsplit version has lower priority
    # とは: と + は
    2028950: ([1008490, 2028920], -5),
    # では: で + は
    1008450: ([2028980, 2028920], -5),
    # だから: だ + から
    1007310: ([2089020, 1002980], -5),
    # 今日は: 今日 + は (greeting/topic marker should split)
    1289400: ([1186220, 2028920], -30),
}

# All split seqs for quick lookup
ALL_SPLIT_SEQS: Set[int] = (
    set(DE_SPLITS.keys()) |
    set(TOORI_SPLITS.keys()) |
    set(DO_SPLITS.keys()) |
    set(SHI_SPLITS.keys()) |
    set(COMPLEX_SPLITS.keys()) |
    set(SEGMENT_SPLITS.keys())
)


# ============================================================================
# Text-based Split Detection (for dictionary-free lookup)
# ============================================================================

# Words ending in で that should be split
DE_SUFFIX_WORDS: Set[str] = {
    '一人で', '何で', 'これで', 'その上で', '無断で', '空で',
    '土足で', '何用で', '交代で', '私費で', '人前で', '全体で',
    '単独で', '道理で', '半眼で', '別封で', '無しで', '名義で',
    '力尽くで', '抜き足で', '金ずくで', '水入らずで', '陰で',
    '差しで', 'もう少しで', '今までで', 'ところで', 'お陰で',
}

# Words ending in 通り that should be split
TOORI_SUFFIX_WORDS: Set[str] = {
    '元通り', '大通り', '中通り', '型通り', '表通り', '本通り',
    '目通り', '人通り', '裏通り', '素通り', '一通り', '二通り',
}

# Words with ど- prefix that should be split
DO_PREFIX_WORDS: Set[str] = {
    'ど下手', 'どすけべ', 'ど根性', 'ど田舎',
}

# Words with し- prefix (する stem) that should be split
SHI_PREFIX_WORDS: Set[str] = {
    'し易い', 'しやすい', 'し吹く', 'し難い', 'しにくい',
    'し過ぎる', 'しすぎる', 'し合う', 'しあう', 'し尽す',
    'し続ける', 'しつづける', 'し兼ねる', 'しかねる',
    'し始める', 'しはじめる', 'し出す', 'しだす',
    'し直す', 'しなおす', 'し慣れる', 'しなれる',
    'し損なう', 'しそこなう', 'し損じる', 'しそんじる',
    'し遂げる', 'しとげる', 'し向ける', 'しむける',
    'し終える', 'しおえる', 'し残す', 'しのこす',
    'し掛ける', 'しかける', 'し損ねる', 'しそこねる',
}

# Complex pattern words
COMPLEX_PATTERN_WORDS: Dict[str, Tuple[List[str], int]] = {
    'なくなる': (['なく', 'なる'], 30),
    '無くなる': (['無く', 'なる'], 30),
    'という': (['と', 'いう'], 20),
    'といった': (['と', 'いった'], 20),
    'といって': (['と', 'いって'], 20),
    'といえば': (['と', 'いえば'], 20),
    'じゃない': (['じゃ', 'ない'], 10),
    'じゃなかった': (['じゃ', 'なかった'], 10),
    'なら': (['なら'], 1),
    '気がつく': (['気', 'が', 'つく'], 100),
    '気がついた': (['気', 'が', 'ついた'], 100),
    '気がつかない': (['気', 'が', 'つかない'], 100),
    '気のせい': (['気', 'の', 'せい'], 100),
}


# ============================================================================
# Split Functions
# ============================================================================

def should_split(surface: str, seq: int) -> bool:
    """
    Check if a word should be split based on its surface form or seq.
    
    Args:
        surface: The surface text of the word
        seq: The JMdict sequence ID
        
    Returns:
        True if the word should be considered for splitting
    """
    # Check by seq first (more accurate)
    if seq in ALL_SPLIT_SEQS:
        return True
    
    # Check by surface pattern
    if surface in DE_SUFFIX_WORDS:
        return True
    if surface in TOORI_SUFFIX_WORDS:
        return True
    if surface in DO_PREFIX_WORDS:
        return True
    if surface in SHI_PREFIX_WORDS:
        return True
    if surface in COMPLEX_PATTERN_WORDS:
        return True
    
    return False


def get_split_score_bonus(surface: str, seq: int) -> int:
    """
    Get the score bonus for splitting a compound word.
    
    Args:
        surface: The surface text of the word
        seq: The JMdict sequence ID
        
    Returns:
        Score bonus (positive = prefer split, negative = prefer whole)
    """
    # Check seq-based splits first
    if seq in DE_SPLITS:
        return DE_SPLITS[seq][1]
    if seq in TOORI_SPLITS:
        return TOORI_SPLITS[seq][2]
    if seq in DO_SPLITS:
        return DO_SPLITS[seq][1]
    if seq in SHI_SPLITS:
        return SHI_SPLITS[seq][1]
    if seq in COMPLEX_SPLITS:
        return COMPLEX_SPLITS[seq][1]
    if seq in SEGMENT_SPLITS:
        return SEGMENT_SPLITS[seq][1]
    
    # Check text-based patterns
    if surface in COMPLEX_PATTERN_WORDS:
        return COMPLEX_PATTERN_WORDS[surface][1]
    
    # Default bonuses by pattern type
    if surface in DE_SUFFIX_WORDS:
        return 20
    if surface in TOORI_SUFFIX_WORDS:
        return 50
    if surface in DO_PREFIX_WORDS:
        return 30
    if surface in SHI_PREFIX_WORDS:
        return 30
    
    return 0


def try_split_word(surface: str, seq: int) -> Optional[SplitResult]:
    """
    Try to split a compound word into its parts.
    
    Args:
        surface: The surface text of the word
        seq: The JMdict sequence ID
        
    Returns:
        SplitResult if the word can be split, None otherwise
    """
    # Try で-ending splits
    if surface.endswith('で') and len(surface) > 1:
        if surface in DE_SUFFIX_WORDS or seq in DE_SPLITS:
            main_text = surface[:-1]
            de_text = 'で'
            
            # Lookup main part
            main_entries = lookup(main_text)
            de_entries = lookup(de_text)
            
            if main_entries and de_entries:
                return SplitResult(
                    parts=[
                        SplitPart(text=main_text, entry=main_entries[0]),
                        SplitPart(text=de_text, entry=de_entries[0]),
                    ],
                    score_bonus=get_split_score_bonus(surface, seq)
                )
    
    # Try 通り-ending splits
    if surface.endswith('通り') and len(surface) > 2:
        if surface in TOORI_SUFFIX_WORDS or seq in TOORI_SPLITS:
            main_text = surface[:-2]
            toori_text = '通り'
            
            main_entries = lookup(main_text)
            toori_entries = lookup(toori_text)
            
            if main_entries and toori_entries:
                return SplitResult(
                    parts=[
                        SplitPart(text=main_text, entry=main_entries[0]),
                        SplitPart(text=toori_text, entry=toori_entries[0]),
                    ],
                    score_bonus=get_split_score_bonus(surface, seq)
                )
    
    # Try ど- prefix splits
    if surface.startswith('ど') and len(surface) > 1:
        if surface in DO_PREFIX_WORDS or seq in DO_SPLITS:
            do_text = 'ど'
            rest_text = surface[1:]
            
            do_entries = lookup(do_text)
            rest_entries = lookup(rest_text)
            
            if do_entries and rest_entries:
                return SplitResult(
                    parts=[
                        SplitPart(text=do_text, entry=do_entries[0]),
                        SplitPart(text=rest_text, entry=rest_entries[0]),
                    ],
                    score_bonus=get_split_score_bonus(surface, seq)
                )
    
    # Try し- prefix splits
    if surface.startswith('し') and len(surface) > 1:
        if surface in SHI_PREFIX_WORDS or seq in SHI_SPLITS:
            shi_text = 'し'
            rest_text = surface[1:]
            
            shi_entries = lookup(shi_text)
            rest_entries = lookup(rest_text)
            
            if shi_entries and rest_entries:
                return SplitResult(
                    parts=[
                        SplitPart(text=shi_text, entry=shi_entries[0]),
                        SplitPart(text=rest_text, entry=rest_entries[0]),
                    ],
                    score_bonus=get_split_score_bonus(surface, seq)
                )
    
    # Try complex patterns
    if surface in COMPLEX_PATTERN_WORDS:
        pattern_parts, score = COMPLEX_PATTERN_WORDS[surface]
        parts = []
        
        for part_text in pattern_parts:
            entries = lookup(part_text)
            if entries:
                parts.append(SplitPart(text=part_text, entry=entries[0]))
            else:
                # If any part not found, return None
                return None
        
        if parts:
            return SplitResult(parts=parts, score_bonus=score)
    
    return None


def get_split_parts(surface: str, seq: int) -> Optional[List[str]]:
    """
    Get the text parts of a split compound word.
    
    Simple helper that returns just the text parts without full lookup.
    
    Args:
        surface: The surface text of the word
        seq: The JMdict sequence ID
        
    Returns:
        List of part texts if split possible, None otherwise
    """
    result = try_split_word(surface, seq)
    if result:
        return [p.text for p in result.parts]
    return None


# ============================================================================
# Score Integration
# ============================================================================

def calculate_split_score_adjustment(surface: str, seq: int) -> float:
    """
    Calculate how much to adjust a word's score if it should be split.
    
    This is used in the tokenizer to prefer or discourage certain splits.
    Positive = prefer the compound word as-is
    Negative = prefer splitting it
    
    Args:
        surface: The surface text of the word
        seq: The JMdict sequence ID
        
    Returns:
        Score adjustment (can be positive or negative)
    """
    if not should_split(surface, seq):
        return 0.0
    
    # For compound words that should be split, we reduce their score
    # to encourage the tokenizer to find the split version
    bonus = get_split_score_bonus(surface, seq)
    
    # Positive bonus means "prefer split" -> negative adjustment to compound
    # Negative bonus means "prefer compound" -> positive adjustment (rare)
    return -bonus


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    'SplitPart',
    'SplitResult',
    'should_split',
    'get_split_score_bonus',
    'try_split_word',
    'get_split_parts',
    'calculate_split_score_adjustment',
    'DE_SPLITS',
    'TOORI_SPLITS', 
    'DO_SPLITS',
    'SHI_SPLITS',
    'COMPLEX_SPLITS',
    'SEGMENT_SPLITS',
    'ALL_SPLIT_SEQS',
]
