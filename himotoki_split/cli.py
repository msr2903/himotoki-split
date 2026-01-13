"""
CLI interface for himotoki-split.

Usage:
    himotoki-split "今日は天気がいいです"
    himotoki-split -d "食べたかった"
    himotoki-split --json "食べました"
"""

import argparse
import json
import sys
from typing import List, Optional, Tuple

from himotoki_split import tokenize, Token, __version__
from himotoki_split.dictionary import lookup, load_dictionary, get_pos_name, get_kana_reading
from himotoki_split.constants import CONJ_TYPE_NAMES


# ============================================================================
# POS Full Names
# ============================================================================

POS_NAMES = {
    'v1': 'Ichidan verb',
    'v1-s': 'Ichidan verb (kureru)',
    'v5k': 'Godan verb (く)',
    'v5k-s': 'Godan verb (行く)',
    'v5g': 'Godan verb (ぐ)',
    'v5s': 'Godan verb (す)',
    'v5t': 'Godan verb (つ)',
    'v5n': 'Godan verb (ぬ)',
    'v5b': 'Godan verb (ぶ)',
    'v5m': 'Godan verb (む)',
    'v5r': 'Godan verb (る)',
    'v5r-i': 'Godan verb (irregular る)',
    'v5u': 'Godan verb (う)',
    'v5u-s': 'Godan verb (う, special)',
    'vs': 'Suru verb',
    'vs-s': 'Suru verb (special)',
    'vs-i': 'Suru verb (included)',
    'vk': 'Kuru verb',
    'adj-i': 'I-adjective',
    'adj-na': 'Na-adjective',
    'adj-ix': 'I-adjective (いい)',
    'n': 'Noun',
    'prt': 'Particle',
    'adv': 'Adverb',
    'conj': 'Conjunction',
    'aux': 'Auxiliary',
    'aux-v': 'Auxiliary verb',
    'aux-adj': 'Auxiliary adjective',
    'cop': 'Copula',
    'exp': 'Expression',
    'int': 'Interjection',
    'pn': 'Pronoun',
    'ctr': 'Counter',
    'suf': 'Suffix',
    'pref': 'Prefix',
}


# ============================================================================
# Conjugation Pattern Detection
# ============================================================================

def detect_conjugation_layers(surface: str, pos: str) -> List[dict]:
    """
    Detect conjugation layers from the surface form.
    
    Analyzes the verb/adjective ending to identify all conjugation layers
    applied to the base form.
    
    Returns list of layers, ordered from innermost to outermost transformation.
    """
    layers = []
    
    # Work backwards from the surface form to detect layers
    # Order matters - check longer/more specific patterns first
    
    # =====================================
    # Causative + Polite patterns (させ)
    # =====================================
    
    # させません (causative + polite + negative)
    if 'させません' in surface:
        layers.append({'name': 'causative', 'code': 'SASE', 'meaning': 'make/let do'})
        layers.append({'name': 'polite', 'code': 'MASU', 'meaning': 'polite'})
        layers.append({'name': 'negative', 'code': 'N', 'meaning': 'not'})
        return layers
    
    # させました (causative + polite + past)
    if 'させました' in surface:
        layers.append({'name': 'causative', 'code': 'SASE', 'meaning': 'make/let do'})
        layers.append({'name': 'polite', 'code': 'MASU', 'meaning': 'polite'})
        layers.append({'name': 'past', 'code': 'TA', 'meaning': 'did'})
        return layers
    
    # させます (causative + polite)
    if 'させます' in surface:
        layers.append({'name': 'causative', 'code': 'SASE', 'meaning': 'make/let do'})
        layers.append({'name': 'polite', 'code': 'MASU', 'meaning': 'polite'})
        return layers
    
    # させられる (causative + passive)
    if 'させられる' in surface or 'させられ' in surface:
        layers.append({'name': 'causative', 'code': 'SASE', 'meaning': 'make/let do'})
        layers.append({'name': 'passive', 'code': 'RARERU', 'meaning': 'is made to'})
        return layers
    
    # させない (causative + negative)
    if 'させない' in surface:
        layers.append({'name': 'causative', 'code': 'SASE', 'meaning': 'make/let do'})
        layers.append({'name': 'negative', 'code': 'NAI', 'meaning': 'not'})
        return layers
    
    # させた (causative + past)
    if surface.endswith('させた'):
        layers.append({'name': 'causative', 'code': 'SASE', 'meaning': 'make/let do'})
        layers.append({'name': 'past', 'code': 'TA', 'meaning': 'did'})
        return layers
    
    # させて (causative + te-form)
    if surface.endswith('させて'):
        layers.append({'name': 'causative', 'code': 'SASE', 'meaning': 'make/let do'})
        layers.append({'name': 'te-form', 'code': 'TE', 'meaning': 'and...'})
        return layers
    
    # させる (causative only)
    if surface.endswith('させる'):
        layers.append({'name': 'causative', 'code': 'SASE', 'meaning': 'make/let do'})
        return layers
    
    # =====================================
    # Godan causative patterns (せる/す)
    # =====================================
    
    # せません (godan causative + polite + negative)
    if surface.endswith('せません') and not surface.endswith('させません'):
        layers.append({'name': 'causative', 'code': 'SE', 'meaning': 'make/let do'})
        layers.append({'name': 'polite', 'code': 'MASU', 'meaning': 'polite'})
        layers.append({'name': 'negative', 'code': 'N', 'meaning': 'not'})
        return layers
    
    # =====================================
    # Passive/Potential patterns (られ/れ)
    # =====================================
    
    # られなかった (passive/potential + negative + past)
    if 'られなかった' in surface or 'れなかった' in surface:
        layers.append({'name': 'passive/potential', 'code': 'RARERU', 'meaning': 'is done/can do'})
        layers.append({'name': 'negative', 'code': 'NAI', 'meaning': 'not'})
        layers.append({'name': 'past', 'code': 'TA', 'meaning': 'did'})
        return layers
    
    # られません (passive/potential + polite + negative)
    if 'られません' in surface:
        layers.append({'name': 'passive/potential', 'code': 'RARERU', 'meaning': 'is done/can do'})
        layers.append({'name': 'polite', 'code': 'MASU', 'meaning': 'polite'})
        layers.append({'name': 'negative', 'code': 'N', 'meaning': 'not'})
        return layers
    
    # られました (passive/potential + polite + past)
    if 'られました' in surface:
        layers.append({'name': 'passive/potential', 'code': 'RARERU', 'meaning': 'is done/can do'})
        layers.append({'name': 'polite', 'code': 'MASU', 'meaning': 'polite'})
        layers.append({'name': 'past', 'code': 'TA', 'meaning': 'did'})
        return layers
    
    # られます (passive/potential + polite)
    if 'られます' in surface:
        layers.append({'name': 'passive/potential', 'code': 'RARERU', 'meaning': 'is done/can do'})
        layers.append({'name': 'polite', 'code': 'MASU', 'meaning': 'polite'})
        return layers
    
    # られない (passive/potential + negative)
    if surface.endswith('られない') or surface.endswith('れない'):
        layers.append({'name': 'passive/potential', 'code': 'RARERU', 'meaning': 'is done/can do'})
        layers.append({'name': 'negative', 'code': 'NAI', 'meaning': 'not'})
        return layers
    
    # られた (passive/potential + past)
    if surface.endswith('られた') or surface.endswith('れた'):
        layers.append({'name': 'passive/potential', 'code': 'RARERU', 'meaning': 'is done/can do'})
        layers.append({'name': 'past', 'code': 'TA', 'meaning': 'did'})
        return layers
    
    # られる / れる (passive/potential only)
    if surface.endswith('られる') or (surface.endswith('れる') and pos.startswith('v5')):
        layers.append({'name': 'passive/potential', 'code': 'RARERU', 'meaning': 'is done/can do'})
        return layers
    
    # =====================================
    # Desiderative patterns (たい)
    # =====================================
    
    # たかった (desiderative + past)
    if surface.endswith('たかった'):
        layers.append({'name': 'desiderative', 'code': 'TAI', 'meaning': 'want to'})
        layers.append({'name': 'past', 'code': 'TA', 'meaning': 'did'})
        return layers
    
    # たくなかった (desiderative + negative + past)
    if surface.endswith('たくなかった'):
        layers.append({'name': 'desiderative', 'code': 'TAI', 'meaning': 'want to'})
        layers.append({'name': 'negative', 'code': 'NAI', 'meaning': 'not'})
        layers.append({'name': 'past', 'code': 'TA', 'meaning': 'did'})
        return layers
    
    # たくない (desiderative + negative)
    if surface.endswith('たくない'):
        layers.append({'name': 'desiderative', 'code': 'TAI', 'meaning': 'want to'})
        layers.append({'name': 'negative', 'code': 'NAI', 'meaning': 'not'})
        return layers
    
    # たい (desiderative only)
    if surface.endswith('たい'):
        layers.append({'name': 'desiderative', 'code': 'TAI', 'meaning': 'want to'})
        return layers
    
    # =====================================
    # Polite patterns (ます)
    # =====================================
    
    # ませんでした (polite + negative + past)
    if surface.endswith('ませんでした'):
        layers.append({'name': 'polite', 'code': 'MASU', 'meaning': 'polite'})
        layers.append({'name': 'negative', 'code': 'N', 'meaning': 'not'})
        layers.append({'name': 'past', 'code': 'TA', 'meaning': 'did'})
        return layers
    
    # ません (polite + negative)
    if surface.endswith('ません'):
        layers.append({'name': 'polite', 'code': 'MASU', 'meaning': 'polite'})
        layers.append({'name': 'negative', 'code': 'N', 'meaning': 'not'})
        return layers
    
    # ました (polite + past)
    if surface.endswith('ました'):
        layers.append({'name': 'polite', 'code': 'MASU', 'meaning': 'polite'})
        layers.append({'name': 'past', 'code': 'TA', 'meaning': 'did'})
        return layers
    
    # ます (polite only)
    if surface.endswith('ます'):
        layers.append({'name': 'polite', 'code': 'MASU', 'meaning': 'polite'})
        return layers
    
    # =====================================
    # Progressive patterns (ている)
    # =====================================
    
    # ていた / てた (progressive + past)
    if surface.endswith('ていた') or surface.endswith('てた'):
        layers.append({'name': 'progressive', 'code': 'TE_IRU', 'meaning': 'is doing'})
        layers.append({'name': 'past', 'code': 'TA', 'meaning': 'did'})
        return layers
    
    # ていない (progressive + negative)
    if surface.endswith('ていない') or surface.endswith('てない'):
        layers.append({'name': 'progressive', 'code': 'TE_IRU', 'meaning': 'is doing'})
        layers.append({'name': 'negative', 'code': 'NAI', 'meaning': 'not'})
        return layers
    
    # ている / てる (progressive only)
    if surface.endswith('ている') or surface.endswith('てる'):
        layers.append({'name': 'progressive', 'code': 'TE_IRU', 'meaning': 'is doing'})
        return layers
    
    # =====================================
    # Simple negative patterns
    # =====================================
    
    # なかった (negative + past)
    if surface.endswith('なかった'):
        layers.append({'name': 'negative', 'code': 'NAI', 'meaning': 'not'})
        layers.append({'name': 'past', 'code': 'TA', 'meaning': 'did'})
        return layers
    
    # ない (negative only)
    if surface.endswith('ない') and (pos.startswith('v') or pos.startswith('adj')):
        layers.append({'name': 'negative', 'code': 'NAI', 'meaning': 'not'})
        return layers
    
    # =====================================
    # Te-form patterns
    # =====================================
    
    if surface.endswith('て') or surface.endswith('で'):
        layers.append({'name': 'te-form', 'code': 'TE', 'meaning': 'and...'})
        return layers
    
    # =====================================
    # Past/ta-form patterns
    # =====================================
    
    if (surface.endswith('た') or surface.endswith('だ')) and pos.startswith('v'):
        layers.append({'name': 'past', 'code': 'TA', 'meaning': 'did'})
        return layers
    
    # =====================================
    # Volitional patterns
    # =====================================
    
    if surface.endswith('よう') or surface.endswith('おう'):
        layers.append({'name': 'volitional', 'code': 'OU', 'meaning': "let's/will"})
        return layers
    
    # =====================================
    # Conditional patterns
    # =====================================
    
    if surface.endswith('れば') or surface.endswith('えば'):
        layers.append({'name': 'conditional', 'code': 'EBA', 'meaning': 'if'})
        return layers
    
    if surface.endswith('たら'):
        layers.append({'name': 'conditional', 'code': 'TARA', 'meaning': 'if/when'})
        return layers
    
    # =====================================
    # Imperative patterns
    # =====================================
    
    if surface.endswith('ろ') or surface.endswith('よ'):
        if pos.startswith('v1'):
            layers.append({'name': 'imperative', 'code': 'RO', 'meaning': 'do!'})
            return layers
    
    # No special conjugation detected
    return layers


def get_base_form_and_reading(surface: str, pos: str, base_seq: int) -> Tuple[str, Optional[str]]:
    """
    Reconstruct the base form and get its reading.
    
    Returns (base_form, base_reading)
    """
    load_dictionary()
    
    # Get kana reading for base form from dictionary
    base_reading = get_kana_reading(base_seq) if base_seq else None
    
    base_form = surface  # Default
    
    # Reconstruct base form based on POS and conjugation patterns
    if pos.startswith('v1'):  # Ichidan verb
        # Try to extract the verb stem and add る
        stem = None
        
        # Causative patterns
        if 'させ' in surface:
            # Find stem before させ
            idx = surface.find('させ')
            if idx > 0:
                stem = surface[:idx]
        # Passive/potential patterns
        elif 'られ' in surface:
            idx = surface.find('られ')
            if idx > 0:
                stem = surface[:idx]
        # Desiderative patterns (remove たい/たかった/たくない)
        elif surface.endswith('たかった'):
            stem = surface[:-4]
        elif surface.endswith('たくない'):
            stem = surface[:-4]
        elif surface.endswith('たい'):
            stem = surface[:-2]
        # Polite patterns
        elif surface.endswith('ませんでした'):
            stem = surface[:-6]
        elif surface.endswith('ません'):
            stem = surface[:-3]
        elif surface.endswith('ました'):
            stem = surface[:-3]
        elif surface.endswith('ます'):
            stem = surface[:-2]
        # Negative patterns
        elif surface.endswith('なかった'):
            stem = surface[:-4]
        elif surface.endswith('ない'):
            stem = surface[:-2]
        # Progressive patterns
        elif surface.endswith('ている'):
            stem = surface[:-3]
        elif surface.endswith('てる'):
            stem = surface[:-2]
        # Te-form
        elif surface.endswith('て'):
            stem = surface[:-1]
        # Past
        elif surface.endswith('た'):
            stem = surface[:-1]
        
        if stem and not stem.endswith('る'):
            base_form = stem + 'る'
    
    # Try to look up the actual base form we've reconstructed
    # This is important for cases where the dictionary stores a conjugated form
    if base_form != surface:
        entries = lookup(base_form)
        if entries:
            actual_reading = get_kana_reading(entries[0].seq)
            if actual_reading:
                return base_form, actual_reading
    
    # Fall back to the stored base_reading if available
    if base_reading:
        # If base_reading is for a conjugated form (e.g., たべられる),
        # and we've reconstructed a dictionary form (e.g., 食べる),
        # try to derive the correct reading
        if base_reading.endswith('られる') and base_form.endswith('る'):
            # base_reading is potential/passive, derive dictionary form reading
            dict_reading = base_reading[:-3] + 'る'  # たべられる -> たべる
            return base_form, dict_reading
        return base_form, base_reading
    
    return base_form, None


def derive_surface_reading(surface: str, base_reading: Optional[str]) -> Optional[str]:
    """
    Derive the reading of the surface form from the base reading.
    """
    if not base_reading:
        return None
    
    # Check patterns from longest to shortest
    # These transformations show how the base reading changes
    
    # Special case: if base_reading already ends with られる (potential/passive form)
    # and surface contains られ, we need to adjust
    if base_reading.endswith('られる'):
        base_stem = base_reading[:-3]  # Remove られる
        
        if surface.endswith('られなかった'):
            return base_stem + 'られなかった'
        if surface.endswith('られない'):
            return base_stem + 'られない'
        if surface.endswith('られた'):
            return base_stem + 'られた'
        if surface.endswith('られました'):
            return base_stem + 'られました'
        if surface.endswith('られません'):
            return base_stem + 'られません'
        if surface.endswith('られます'):
            return base_stem + 'られます'
        if surface.endswith('られる'):
            return base_reading  # Already correct
    
    # Causative + polite patterns
    if 'させません' in surface and base_reading.endswith('る'):
        return base_reading[:-1] + 'させません'
    if 'させました' in surface and base_reading.endswith('る'):
        return base_reading[:-1] + 'させました'
    if 'させます' in surface and base_reading.endswith('る'):
        return base_reading[:-1] + 'させます'
    if surface.endswith('させて') and base_reading.endswith('る'):
        return base_reading[:-1] + 'させて'
    if surface.endswith('させる') and base_reading.endswith('る'):
        return base_reading[:-1] + 'させる'
    
    # Passive/potential patterns
    if surface.endswith('られなかった') and base_reading.endswith('る'):
        return base_reading[:-1] + 'られなかった'
    if surface.endswith('られない') and base_reading.endswith('る'):
        return base_reading[:-1] + 'られない'
    if surface.endswith('られた') and base_reading.endswith('る'):
        return base_reading[:-1] + 'られた'
    if surface.endswith('られる') and base_reading.endswith('る'):
        return base_reading[:-1] + 'られる'
    
    # Desiderative patterns
    if surface.endswith('たかった') and base_reading.endswith('る'):
        return base_reading[:-1] + 'たかった'
    if surface.endswith('たくない') and base_reading.endswith('る'):
        return base_reading[:-1] + 'たくない'
    if surface.endswith('たい') and base_reading.endswith('る'):
        return base_reading[:-1] + 'たい'
    
    # Polite patterns
    if surface.endswith('ませんでした') and base_reading.endswith('る'):
        return base_reading[:-1] + 'ませんでした'
    if surface.endswith('ません') and base_reading.endswith('る'):
        return base_reading[:-1] + 'ません'
    if surface.endswith('ました') and base_reading.endswith('る'):
        return base_reading[:-1] + 'ました'
    if surface.endswith('ます') and base_reading.endswith('る'):
        return base_reading[:-1] + 'ます'
    
    # Negative patterns
    if surface.endswith('なかった') and base_reading.endswith('る'):
        return base_reading[:-1] + 'なかった'
    if surface.endswith('ない') and base_reading.endswith('る'):
        return base_reading[:-1] + 'ない'
    
    # Progressive patterns
    if surface.endswith('ている') and base_reading.endswith('る'):
        return base_reading[:-1] + 'ている'
    if surface.endswith('てる') and base_reading.endswith('る'):
        return base_reading[:-1] + 'てる'
    
    # Te-form
    if surface.endswith('て') and base_reading.endswith('る'):
        return base_reading[:-1] + 'て'
    
    # Past
    if surface.endswith('た') and base_reading.endswith('る'):
        return base_reading[:-1] + 'た'
    
    return base_reading


def fix_contextual_reading(surface: str, reading: Optional[str], next_surface: Optional[str]) -> Optional[str]:
    """
    Fix readings based on context.
    
    Some words have multiple readings depending on context:
    - 何: なに before を/が/も/か, なん before で/と/の/だ/です
    """
    if not reading:
        return reading
    
    # 何 reading depends on following particle
    if surface == '何':
        if next_surface in ('を', 'が', 'も', 'か', 'に', 'へ'):
            return 'なに'
        elif next_surface in ('で', 'と', 'の', 'だ', 'です', 'て', 'ですか'):
            return 'なん'
        # Default to なに for question contexts
        if next_surface is None or next_surface in ('？', '?'):
            return 'なに'
    
    # 来る (kuru) irregular verb - conjugations change reading
    # くる → きた (past), きて (te-form), きます (polite), etc.
    if reading == 'くる' or (reading and reading.startswith('く')):
        if surface == '来た':
            return 'きた'
        if surface == '来て':
            return 'きて'
        if surface == '来ます':
            return 'きます'
        if surface == '来ました':
            return 'きました'
        if surface == '来ない':
            return 'こない'
        if surface == '来なかった':
            return 'こなかった'
        if surface == '来る':
            return 'くる'
        if surface == '来られる':
            return 'こられる'
        if surface == '来い':
            return 'こい'
    
    return reading


# ============================================================================
# Output Formatting
# ============================================================================

def format_default(tokens: List[Token]) -> str:
    """
    Default output: simple splitting like original himotoki.
    
    Just shows: surface1 | surface2 | surface3
    """
    surfaces = [t.surface for t in tokens]
    return " | ".join(surfaces)


def format_detailed(tokens: List[Token]) -> str:
    """
    Detailed output with conjugation breakdown.
    
    Compact vertical format showing all tokens with readings.
    """
    load_dictionary()
    
    lines = []
    
    # First, show the full split
    surfaces = [t.surface for t in tokens]
    lines.append(" | ".join(surfaces))
    lines.append("─" * 40)
    
    for i, t in enumerate(tokens):
        surface = t.surface
        pos = t.pos
        base_seq = t.base_form_id
        
        # Get next token surface for context-based reading
        next_surface = tokens[i + 1].surface if i + 1 < len(tokens) else None
        
        # Get base form and reading
        base_form, base_reading = get_base_form_and_reading(surface, pos, base_seq)
        
        # Get surface reading
        surface_reading = derive_surface_reading(surface, base_reading)
        
        # Fix contextual readings (e.g., 何 → なに/なん)
        surface_reading = fix_contextual_reading(surface, surface_reading, next_surface)
        base_reading = fix_contextual_reading(surface, base_reading, next_surface)
        
        # Get conjugation layers
        layers = detect_conjugation_layers(surface, pos)
        
        # Get POS name
        pos_name = POS_NAMES.get(pos, pos)
        
        # Build compact single-line or two-line output per token
        # Format: surface【reading】 (POS) ← base【reading】
        parts = []
        
        # Surface with reading
        if surface_reading and surface_reading != surface:
            parts.append(f"{surface}【{surface_reading}】")
        else:
            parts.append(surface)
        
        # POS in parentheses
        parts.append(f"({pos_name})")
        
        # Base form if different
        if base_form != surface:
            if base_reading:
                parts.append(f"← {base_form}【{base_reading}】")
            else:
                parts.append(f"← {base_form}")
        
        lines.append(" ".join(parts))
        
        # Conjugation layers on next line if present
        if layers:
            layer_parts = []
            for layer in layers:
                layer_parts.append(f"{layer['name']}({layer['code']})")
            lines.append("  └─ " + " → ".join(layer_parts))
    
    return "\n".join(lines)


def format_json(tokens: List[Token]) -> str:
    """Format tokens as JSON with full details."""
    load_dictionary()
    
    data = []
    for t in tokens:
        base_form, base_reading = get_base_form_and_reading(t.surface, t.pos, t.base_form_id)
        surface_reading = derive_surface_reading(t.surface, base_reading)
        layers = detect_conjugation_layers(t.surface, t.pos)
        
        data.append({
            "surface": t.surface,
            "reading": surface_reading,
            "pos": t.pos,
            "pos_name": POS_NAMES.get(t.pos, t.pos),
            "base_form": base_form,
            "base_reading": base_reading,
            "base_form_id": t.base_form_id,
            "start": t.start,
            "end": t.end,
            "conjugation_layers": layers,
        })
    
    return json.dumps(data, ensure_ascii=False, indent=2)


def format_simple(tokens: List[Token]) -> str:
    """Simple tab-separated output format."""
    lines = []
    for t in tokens:
        lines.append(f"{t.surface}\t{t.base_form}\t{t.pos}\t{t.base_form_id}")
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        prog="himotoki-split",
        description="Lightweight Japanese Morphological Analyzer",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Japanese text to tokenize",
    )
    parser.add_argument(
        "--detail", "-d",
        action="store_true",
        help="Show detailed conjugation breakdown",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--simple", "-s",
        action="store_true",
        help="Simple output format (surface, base, pos, id)",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"himotoki-split {__version__}",
    )
    
    args = parser.parse_args()
    
    if args.text is None:
        # Read from stdin
        text = sys.stdin.read().strip()
    else:
        text = args.text
    
    if not text:
        parser.print_help()
        sys.exit(1)
    
    try:
        tokens = tokenize(text)
        
        if args.json:
            print(format_json(tokens))
        elif args.simple:
            print(format_simple(tokens))
        elif args.detail:
            print(format_detailed(tokens))
        else:
            # Default: simple splitting output
            print(format_default(tokens))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
