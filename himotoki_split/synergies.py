"""
Synergy and penalty scoring module for himotoki-split.
Ported from original himotoki's synergies.py.

Synergies are score bonuses applied to adjacent token pairs that commonly
appear together (e.g., noun + particle, na-adjective + な).

Penalties are score reductions for combinations that should be avoided
(e.g., と + は when compound とは exists).

Since himotoki-split doesn't have database access, this module uses
pre-computed rules based on POS, surface forms, and sequence IDs.
"""

from typing import List, Tuple, Optional, Set, Dict, Callable, Any
from dataclasses import dataclass

from himotoki_split.constants import (
    SEQ_WA, SEQ_GA, SEQ_NI, SEQ_DE, SEQ_TO, SEQ_MO, SEQ_WO, SEQ_NO,
    SEQ_TOHA, SEQ_NIHA, SEQ_NITSURE, SEQ_OSUSUME,
    NOUN_PARTICLES,
)


# ============================================================================
# Synergy Data Structures
# ============================================================================

@dataclass(slots=True)
class Synergy:
    """A synergy bonus between two tokens."""
    description: str    # Human-readable description
    score: float        # Score bonus/penalty
    connector: str      # Connector between tokens (for display)


# ============================================================================
# Filter Functions (for matching token properties)
# ============================================================================

# POS IDs that are nouns (from dictionary.py POS_ID_MAP)
NOUN_POS_IDS: Set[int] = {1, 2, 3, 4, 5}  # n, n-adv, n-pref, n-suf, n-t

# POS IDs that are particles
PARTICLE_POS_IDS: Set[int] = {82}  # prt

# POS IDs that are na-adjectives
NA_ADJ_POS_IDS: Set[int] = {42}  # adj-na

# POS IDs that are no-adjectives  
NO_ADJ_POS_IDS: Set[int] = {43}  # adj-no

# POS IDs that are adverbs
ADV_POS_IDS: Set[int] = {50, 51}  # adv, adv-to

# POS IDs that are counters
COUNTER_POS_IDS: Set[int] = {72}  # ctr

# POS IDs that are vs (suru verbs)
VS_POS_IDS: Set[int] = {27, 28, 29}  # vs, vs-i, vs-s


def is_noun(pos_id: int) -> bool:
    """Check if POS ID represents a noun."""
    return pos_id in NOUN_POS_IDS


def is_particle(pos_id: int) -> bool:
    """Check if POS ID represents a particle."""
    return pos_id in PARTICLE_POS_IDS


def is_na_adj(pos_id: int) -> bool:
    """Check if POS ID represents a na-adjective."""
    return pos_id in NA_ADJ_POS_IDS


def is_counter(pos_id: int) -> bool:
    """Check if POS ID represents a counter."""
    return pos_id in COUNTER_POS_IDS


# ============================================================================
# Synergy Rules (Score Bonuses)
# ============================================================================

# Particles that commonly follow nouns (seq IDs)
NOUN_PARTICLE_SEQS: Set[int] = NOUN_PARTICLES


def get_synergy_bonus(
    left_surface: str,
    left_pos_id: int,
    left_seq: int,
    right_surface: str,
    right_pos_id: int,
    right_seq: int,
) -> Optional[Synergy]:
    """
    Calculate synergy bonus between two adjacent tokens.
    
    Returns:
        Synergy object if a bonus applies, None otherwise
    """
    
    # noun + particle synergy (but not と + は)
    if is_noun(left_pos_id) and right_seq in NOUN_PARTICLE_SEQS:
        # Exclude と + は (prefer compound とは)
        if left_seq == SEQ_TO and right_seq == SEQ_WA:
            return None
        score = 10 + 4 * len(right_surface)
        return Synergy("noun+prt", score, " ")
    
    # noun + だ
    if is_noun(left_pos_id) and right_surface == 'だ':
        return Synergy("noun+da", 10, " ")
    
    # の + だ/です/なんだ
    if left_surface in ('の', 'ん') and right_surface in ('だ', 'です', 'だった', 'だろう', 'なんだ'):
        return Synergy("no+da", 15, " ")
    
    # そう + なんだ
    if left_surface == 'そう' and right_surface == 'なんだ':
        return Synergy("sou+nanda", 50, " ")
    
    # na-adjective + な/に
    if is_na_adj(left_pos_id) and right_surface in ('な', 'に'):
        return Synergy("na-adj", 15, " ")
    
    # no-adjective + の
    if left_pos_id in NO_ADJ_POS_IDS and right_surface == 'の':
        return Synergy("no-adj", 15, " ")
    
    # to-adverb + と
    if left_pos_id == 51 and right_surface == 'と':  # adv-to
        score = 10 + 10 * len(left_surface)
        return Synergy("adv-to", score, " ")
    
    # noun + 中
    if is_noun(left_pos_id) and right_surface == '中':
        return Synergy("suffix-chu", 12, "-")
    
    # noun + たち
    if is_noun(left_pos_id) and right_surface == 'たち':
        return Synergy("suffix-tachi", 10, "-")
    
    # noun + ぶり
    if is_noun(left_pos_id) and right_surface == 'ぶり':
        return Synergy("suffix-buri", 40, "")
    
    # noun + 性
    if is_noun(left_pos_id) and right_surface == '性':
        return Synergy("suffix-sei", 12, "")
    
    # お/ご + noun (polite prefix)
    if left_surface in ('お', 'ご') and is_noun(right_pos_id):
        # Exclude ご + みの (prefer compound ごみ)
        if left_surface == 'ご' and right_surface in ('みの', 'み'):
            return None
        return Synergy("o+noun", 10, "")
    
    # 未/不 prefix + noun
    if left_surface in ('未', '不') and is_noun(right_pos_id):
        return Synergy("prefix+noun", 15, "")
    
    # しちゃ/しては + いけない
    if left_surface.endswith('は') and right_surface in ('いけない', 'ならない', 'だめ'):
        return Synergy("shicha+ikenai", 50, " ")
    
    # の + 通り
    if left_surface == 'の' and right_surface == '通り':
        return Synergy("no+toori", 50, " ")
    
    # counter + おき
    if is_counter(left_pos_id) and right_surface in ('おき', '置き'):
        return Synergy("counter+oki", 20, "")
    
    # かどうか + は
    if left_surface == 'かどうか' and right_surface == 'は':
        return Synergy("kadouka+wa", 30, " ")
    
    # しか + negative
    if left_surface == 'しか' and right_surface in ('ない', 'なかった', 'ません', 'ありません'):
        return Synergy("shika+neg", 50, " ")
    
    # verb + なければ + ならない/いけない
    if left_surface.endswith('なければ') and right_surface in ('ならない', 'いけない', 'なりません', 'いけません'):
        return Synergy("nakereba+naranai", 60, " ")
    
    # verb + ても + いい
    if left_surface.endswith('ても') and right_surface in ('いい', 'いいですか', 'いいです'):
        return Synergy("temo+ii", 40, " ")
    
    # verb + ことが + できる/ある
    if left_surface == 'ことが' and right_surface in ('できる', 'できます', 'ある', 'あります'):
        return Synergy("kotoga+dekiru", 50, " ")
    
    # verb + ようになる
    if left_surface == 'ように' and right_surface in ('なる', 'なります', 'なった', 'なりました'):
        return Synergy("youni+naru", 45, " ")
    
    # verb + てしまう patterns
    if left_surface.endswith('て') and right_surface in ('しまう', 'しまった', 'しまいます', 'しまいました'):
        return Synergy("te+shimau", 35, "")
    
    # verb + ておく patterns
    if left_surface.endswith('て') and right_surface in ('おく', 'おいた', 'おきます', 'おきました'):
        return Synergy("te+oku", 35, "")
    
    # verb + てくる patterns
    if left_surface.endswith('て') and right_surface in ('くる', 'きた', 'きます', 'きました', 'こない'):
        return Synergy("te+kuru", 35, "")
    
    # verb + ていく patterns
    if left_surface.endswith('て') and right_surface in ('いく', 'いった', 'いきます', 'いきました'):
        return Synergy("te+iku", 35, "")
    
    # verb + てくれる/もらう patterns
    if left_surface.endswith('て') and right_surface in ('くれる', 'くれた', 'もらう', 'もらった', 'もらえる', 'もらえた'):
        return Synergy("te+kureru", 35, "")
    
    # verb + ている patterns (strong synergy)
    if left_surface.endswith('て') and right_surface in ('いる', 'いた', 'います', 'いました', 'いない', 'いません'):
        return Synergy("te+iru", 50, "")
    
    # Xば + Xほど
    if left_surface.endswith('ば') and right_surface.endswith('ほど'):
        return Synergy("ba+hodo", 30, " ")
    
    # わけ patterns
    if left_surface == 'わけ' and right_surface in ('が', 'は', 'に', 'では', 'だ', 'です'):
        return Synergy("wake+prt", 25, "")
    
    # わけにはいかない
    if left_surface == 'わけには' and right_surface in ('いかない', 'いきません'):
        return Synergy("wakeniha+ikanai", 60, "")
    
    # ことにする/なる
    if left_surface == 'ことに' and right_surface in ('する', 'した', 'します', 'しました', 'なる', 'なった', 'なります', 'なりました'):
        return Synergy("kotoni+suru", 45, "")
    
    # ために
    if left_surface == 'ため' and right_surface == 'に':
        return Synergy("tame+ni", 25, "")
    
    # について/において/によって
    if left_surface == 'に' and right_surface in ('ついて', 'おいて', 'よって', 'とって', '対して', 'たいして'):
        return Synergy("ni+tsuite", 40, "")
    
    # かもしれない
    if left_surface == 'かも' and right_surface in ('しれない', 'しれません', 'しれなかった'):
        return Synergy("kamo+shirenai", 55, "")
    
    # てform + ください patterns
    if left_surface.endswith('て') and right_surface in ('ください', 'くださる', 'くださった', 'くださいます'):
        return Synergy("te+kudasai", 50, "")
    
    # から + といって
    if left_surface == 'から' and right_surface in ('といって', 'といった'):
        return Synergy("kara+toitte", 35, " ")
    
    # のに (despite)
    if left_surface == 'の' and right_surface == 'に' and right_pos_id in PARTICLE_POS_IDS:
        return Synergy("noni", 20, "")
    
    # なくなる (become not)
    if left_surface == 'なく' and right_surface in ('なる', 'なった', 'なります', 'なりました'):
        return Synergy("naku+naru", 40, "")
    
    return None


# ============================================================================
# Penalty Rules (Score Reductions)
# ============================================================================

# Words that should NOT be split (prefer compound)
COMPOUND_PREFERENCE_WORDS = {
    'おすすめ', 'ごみ', 'わかんない', '知らんけど', 'しらんけど',
    'からといって', 'にもかかわらず', 'ということ', 'というのは',
    'わけにはいかない', 'ことができる', 'ようになる',
}


def get_penalty(
    left_surface: str,
    left_pos_id: int,
    left_seq: int,
    right_surface: str,
    right_pos_id: int,
    right_seq: int,
) -> Optional[Synergy]:
    """
    Calculate penalty between two adjacent tokens.
    
    Penalties prevent incorrect splits when a compound word exists.
    
    Returns:
        Synergy object with negative score if penalty applies, None otherwise
    """
    
    # と + は penalty (prefer compound とは)
    if left_surface == 'と' and right_surface == 'は':
        return Synergy("to+wa-penalty", -20, " ")
    
    # に + つれ penalty (prefer compound につれ)
    if left_surface == 'に' and right_surface == 'つれ':
        return Synergy("ni+tsure-penalty", -30, " ")
    
    # お + すすめ penalty (prefer compound おすすめ)
    if left_surface == 'お' and right_surface in ('すすめ', '勧め', '薦め'):
        return Synergy("o+susume-penalty", -40, " ")
    
    # ご + みの penalty (prefer compound ごみ)
    if left_surface == 'ご' and right_surface in ('みの', 'み'):
        return Synergy("go+mino-penalty", -15, " ")
    
    # わかん + ない penalty (prefer compound わかんない)
    if left_surface == 'わかん' and right_surface == 'ない':
        return Synergy("wakan+nai-penalty", -30, " ")
    
    # 知らん + けど penalty (prefer compound 知らんけど)
    if left_surface in ('知らん', 'しらん') and right_surface in ('けど', 'けれど'):
        return Synergy("shiran+kedo-penalty", -100, " ")
    
    # から + とい penalty (prefer compound からといって)
    if left_surface == 'から' and right_surface in ('とい', 'といっ'):
        return Synergy("kara+toi-penalty", -35, " ")
    
    # 人がい + たら penalty (misparse of 人がいい)
    if left_surface == '人がい' and right_surface == 'たら':
        return Synergy("hitogai+tara-penalty", -75, " ")
    
    # 分 + から penalty (misparse of 分かる)
    if left_surface == '分' and right_surface in ('から', 'かる', 'かった', 'かって'):
        return Synergy("bun+kara-penalty", -50, " ")
    
    # 恐れ + 入る penalty (prefer compound 恐れ入る)
    if left_surface == '恐れ' and right_surface in ('入る', '入ります', '入りますが'):
        return Synergy("osore+iru-penalty", -30, " ")
    
    # 待って + くだ penalty (misparse of 待ってください)
    if left_surface == '待って' and right_surface == 'くだ':
        return Synergy("matte+kuda-penalty", -60, " ")
    
    # 信じて + くれ penalty (prefer compound 信じてくれない)
    if left_surface.endswith('て') and right_surface == 'くれ':
        return Synergy("te+kure-penalty", -25, " ")
    
    # 負け + ない penalty (prefer 負けない as compound verb)
    if left_surface == '負け' and right_surface == 'ない':
        return Synergy("make+nai-penalty", -40, " ")
    
    # 経済 + 政策 penalty (prefer compound 経済政策)
    if left_surface == '経済' and right_surface == '政策':
        return Synergy("keizai+seisaku-penalty", -25, " ")
    
    # 調査 + 中 penalty (prefer compound 調査中)
    if is_noun(left_pos_id) and right_surface == '中' and len(left_surface) >= 2:
        # This is actually a synergy, not penalty
        return None
    
    # Short kana words together penalty
    if len(left_surface) == 1 and len(right_surface) == 1:
        # Exclude と, の, は, が which are common single-char particles
        if right_surface not in ('と', 'の', 'は', 'が', 'を', 'に', 'で', 'も'):
            return Synergy("short-penalty", -9, " ")
    
    return None


# ============================================================================
# Main API
# ============================================================================

def apply_synergies(tokens: List[Any]) -> float:
    """
    Calculate total synergy bonus for a list of tokens.
    
    Args:
        tokens: List of token objects (must have surface, pos_id, seq attributes)
    
    Returns:
        Total synergy bonus (can be positive or negative)
    """
    if len(tokens) < 2:
        return 0.0
    
    total_bonus = 0.0
    
    for i in range(len(tokens) - 1):
        left = tokens[i]
        right = tokens[i + 1]
        
        # Get token properties (handle both Token and Segment types)
        left_surface = getattr(left, 'surface', str(left))
        left_pos_id = getattr(left, 'pos_id', 0)
        if hasattr(left, 'entry'):
            left_pos_id = left.entry.pos_id
            left_seq = left.entry.seq
        else:
            left_seq = getattr(left, 'base_form_id', 0)
        
        right_surface = getattr(right, 'surface', str(right))
        right_pos_id = getattr(right, 'pos_id', 0)
        if hasattr(right, 'entry'):
            right_pos_id = right.entry.pos_id
            right_seq = right.entry.seq
        else:
            right_seq = getattr(right, 'base_form_id', 0)
        
        # Check for synergy bonus
        synergy = get_synergy_bonus(
            left_surface, left_pos_id, left_seq,
            right_surface, right_pos_id, right_seq
        )
        if synergy:
            total_bonus += synergy.score
        
        # Check for penalty
        penalty = get_penalty(
            left_surface, left_pos_id, left_seq,
            right_surface, right_pos_id, right_seq
        )
        if penalty:
            total_bonus += penalty.score
    
    return total_bonus


def adjust_path_score(tokens: List[Any], base_score: float) -> float:
    """
    Adjust a path's score with synergy bonuses.
    
    Args:
        tokens: List of tokens in the path
        base_score: Original path score
    
    Returns:
        Adjusted score including synergies
    """
    synergy_bonus = apply_synergies(tokens)
    return base_score + synergy_bonus
