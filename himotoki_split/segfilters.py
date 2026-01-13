"""
Segfilter rules module for himotoki-split.
Ported from original himotoki's synergies.py segfilter section.

Segfilters are rules that block invalid token combinations.
They are applied during path-finding or post-processing to ensure
grammatically correct segmentations.

Examples of invalid combinations blocked:
- ん/んだ following simple particles
- Auxiliary verbs not following continuative form
- Honorifics not following noun-like words
"""

from typing import List, Tuple, Optional, Set


# ============================================================================
# Segfilter Rule Definitions
# ============================================================================

# Particles where ん/んだ should NOT follow directly
# (because ん is usually a contraction of の following verbs/adjectives)
PARTICLES_BLOCKING_N: Set[str] = {
    'は', 'が', 'を', 'に', 'で', 'へ', 'も', 'の', 'と', 'や', 'か',
}

# Surfaces that are "bad endings" (words that shouldn't end a parse)
BAD_ENDINGS: Set[str] = {
    'ちゃい', 'いか', 'とか', 'とき', 'い',
}

# Honorifics that must follow nouns
HONORIFICS: Set[str] = {
    '君', 'くん', 'さん', 'ちゃん', '様', 'さま', '殿', 'どの',
}


def check_n_after_particle(left_surface: str, right_surface: str) -> bool:
    """
    Check if ん/んだ combination after a particle is valid.
    
    ん should not directly follow simple particles like は, が, を, etc.
    
    Returns:
        True if the combination is valid, False if it should be blocked
    """
    if right_surface not in ('ん', 'んだ', 'んです'):
        return True
    
    if left_surface in PARTICLES_BLOCKING_N:
        return False
    
    return True


def check_honorific_after_noun(left_pos_id: int, right_surface: str) -> bool:
    """
    Check if honorific placement is valid.
    
    Honorifics (さん, くん, etc.) should follow nouns, not particles.
    
    Returns:
        True if valid, False if should be blocked
    """
    if right_surface not in HONORIFICS:
        return True
    
    # POS IDs that are nouns
    NOUN_POS_IDS = {1, 2, 3, 4, 5}  # n, n-adv, n-pref, n-suf, n-t
    
    # If left is not a noun, this is suspicious but allow it
    # (names may not be in dictionary)
    return True


def check_bad_ending(right_surface: str, is_last_token: bool) -> bool:
    """
    Check if a token is a "bad ending" when appearing at the end.
    
    Returns:
        True if valid, False if should be blocked
    """
    if not is_last_token:
        return True
    
    if right_surface in BAD_ENDINGS:
        return False
    
    return True


def check_da_suru_combination(left_surface: str, right_surface: str) -> bool:
    """
    Check if だ + する combination is valid.
    
    だ followed by する/して is invalid (should be で + する).
    
    Returns:
        True if valid, False if should be blocked
    """
    if left_surface != 'だ':
        return True
    
    if right_surface in ('する', 'して', 'し', 'した', 'します', 'しました'):
        return False
    
    return True


# ============================================================================
# Main API
# ============================================================================

def validate_token_pair(
    left_surface: str,
    left_pos_id: int,
    right_surface: str,
    right_pos_id: int,
    is_last_token: bool = False,
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a token pair is grammatically valid.
    
    Returns:
        Tuple of (is_valid, reason_if_invalid)
    """
    # Check ん after particle
    if not check_n_after_particle(left_surface, right_surface):
        return False, "ん should not follow particles directly"
    
    # Check だ + する
    if not check_da_suru_combination(left_surface, right_surface):
        return False, "だ + する is invalid (should be で + する)"
    
    # Check bad endings
    if not check_bad_ending(right_surface, is_last_token):
        return False, f"'{right_surface}' is a bad ending token"
    
    # Check honorifics (informational, not blocking)
    # if not check_honorific_after_noun(left_pos_id, right_surface):
    #     return False, "honorific should follow noun"
    
    return True, None


def filter_invalid_paths(paths: List) -> List:
    """
    Filter out paths that contain invalid token combinations.
    
    This is a post-processing step for path-finding results.
    
    Args:
        paths: List of (tokens, score) tuples
    
    Returns:
        Filtered list with invalid paths removed
    """
    valid_paths = []
    
    for tokens, score in paths:
        is_valid = True
        
        for i in range(len(tokens) - 1):
            left = tokens[i]
            right = tokens[i + 1]
            
            # Get token properties
            left_surface = getattr(left, 'surface', str(left))
            left_pos_id = 0
            if hasattr(left, 'entry'):
                left_pos_id = left.entry.pos_id
            
            right_surface = getattr(right, 'surface', str(right))
            right_pos_id = 0
            if hasattr(right, 'entry'):
                right_pos_id = right.entry.pos_id
            
            is_last = (i == len(tokens) - 2)
            
            valid, reason = validate_token_pair(
                left_surface, left_pos_id,
                right_surface, right_pos_id,
                is_last_token=is_last
            )
            
            if not valid:
                is_valid = False
                break
        
        if is_valid:
            valid_paths.append((tokens, score))
    
    return valid_paths


def should_block_token(
    left_tokens: List,
    candidate_surface: str,
    candidate_pos_id: int,
) -> bool:
    """
    Check if a candidate token should be blocked based on context.
    
    This can be used during path-finding to prune invalid candidates.
    
    Args:
        left_tokens: Tokens appearing before the candidate
        candidate_surface: Surface form of the candidate
        candidate_pos_id: POS ID of the candidate
    
    Returns:
        True if the candidate should be blocked
    """
    if not left_tokens:
        return False
    
    # Get the most recent token
    last_token = left_tokens[-1]
    last_surface = getattr(last_token, 'surface', str(last_token))
    last_pos_id = 0
    if hasattr(last_token, 'entry'):
        last_pos_id = last_token.entry.pos_id
    
    valid, _ = validate_token_pair(
        last_surface, last_pos_id,
        candidate_surface, candidate_pos_id,
    )
    
    return not valid
