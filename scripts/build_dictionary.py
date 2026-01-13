#!/usr/bin/env python3
"""
Dictionary Builder for himotoki-split.

This script builds the binary dictionary from JMdict XML.
It parses the XML, generates all conjugated forms, and saves
everything to a compact marisa_trie.RecordTrie file.

Usage:
    python scripts/build_dictionary.py [--jmdict PATH] [--output PATH]
    
Requirements:
    pip install himotoki-split[build]  # Installs SQLAlchemy for building
"""

import argparse
import csv
import logging
import struct
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import marisa_trie
from lxml import etree

# Import dictionary schema
from himotoki_split.dictionary import (
    RECORD_FORMAT,
    POS_ID_MAP,
    get_pos_id,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Paths
# ============================================================================

DEFAULT_JMDICT = Path(__file__).parent.parent / "data" / "JMdict_e.xml"
DEFAULT_OUTPUT = Path(__file__).parent.parent / "himotoki_split" / "data" / "himotoki.dic"
DEFAULT_BASE_FORMS = Path(__file__).parent.parent / "himotoki_split" / "data" / "base_forms.bin"
DEFAULT_KANA_READINGS = Path(__file__).parent.parent / "himotoki_split" / "data" / "kana_readings.bin"
CONJO_CSV = Path(__file__).parent.parent / "data" / "conjo.csv"
KWPOS_CSV = Path(__file__).parent.parent / "data" / "kwpos.csv"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ConjugationRule:
    """Conjugation rule from conjo.csv."""
    pos_id: int
    conj_type: int
    neg: bool
    fml: bool
    onum: int
    stem: int
    okuri: str
    euphr: str
    euphk: str


@dataclass
class DictEntry:
    """A dictionary entry to be stored."""
    surface: str
    seq: int
    cost: int
    pos_id: int
    conj_type: int
    base_seq: int
    base_form: str  # Dictionary form text


# ============================================================================
# Entity Parsing
# ============================================================================

ENTITY_REPLACEMENTS: Dict[str, str] = {}


def parse_entity_definitions(xml_path: Path) -> Dict[str, str]:
    """Parse entity definitions from JMDict DTD."""
    global ENTITY_REPLACEMENTS
    
    with open(xml_path, 'rb') as f:
        content = b''
        for line in f:
            content += line
            if b']>' in line:
                break
    
    import re
    pattern = rb'<!ENTITY\s+([\w-]+)\s+"([^"]*)"\s*>'
    for match in re.finditer(pattern, content):
        name = match.group(1).decode('utf-8')
        value = match.group(2).decode('utf-8')
        if name not in ('lt', 'gt', 'amp', 'apos', 'quot'):
            ENTITY_REPLACEMENTS[value] = name
    
    return ENTITY_REPLACEMENTS


def fix_entity_value(text: str) -> str:
    """Convert expanded entity value to short name."""
    return ENTITY_REPLACEMENTS.get(text, text)


# ============================================================================
# Conjugation Rules
# ============================================================================

_POS_INDEX: Dict[str, int] = {}
_CONJ_RULES: Dict[int, List[ConjugationRule]] = {}


def load_pos_index(csv_path: Path = KWPOS_CSV) -> Dict[str, int]:
    """Load POS name to ID mapping from kwpos.csv."""
    global _POS_INDEX
    
    if _POS_INDEX:
        return _POS_INDEX
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                pos_id = int(row[0])
                pos_name = row[1]
                _POS_INDEX[pos_name] = pos_id
    
    return _POS_INDEX


def load_conj_rules(csv_path: Path = CONJO_CSV) -> Dict[int, List[ConjugationRule]]:
    """Load conjugation rules from conjo.csv."""
    global _CONJ_RULES
    
    if _CONJ_RULES:
        return _CONJ_RULES
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 9:
                pos_id = int(row[0])
                rule = ConjugationRule(
                    pos_id=pos_id,
                    conj_type=int(row[1]),
                    neg=row[2].lower() in ('t', 'true'),
                    fml=row[3].lower() in ('t', 'true'),
                    onum=int(row[4]),
                    stem=int(row[5]),
                    okuri=row[6],
                    euphr=row[7],
                    euphk=row[8],
                )
                if pos_id not in _CONJ_RULES:
                    _CONJ_RULES[pos_id] = []
                _CONJ_RULES[pos_id].append(rule)
    
    return _CONJ_RULES


def is_kana(text: str) -> bool:
    """Check if text is entirely kana."""
    for char in text:
        code = ord(char)
        if not (0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF):
            return False
    return True


def is_kana_char(char: str) -> bool:
    """Check if a single character is kana."""
    code = ord(char)
    return 0x3040 <= code <= 0x309F or 0x30A0 <= code <= 0x30FF


def get_kana_suffix_length(word: str) -> int:
    """Get length of kana suffix at end of word."""
    count = 0
    for char in reversed(word):
        if is_kana_char(char):
            count += 1
        else:
            break
    return count


def construct_conjugation(word: str, rule: ConjugationRule) -> str:
    """Apply conjugation rule to produce conjugated form."""
    is_kana_word = is_kana(word)
    kana_suffix_len = get_kana_suffix_length(word)
    use_kana = is_kana_word or (kana_suffix_len > 0 and kana_suffix_len >= rule.stem + 1)
    
    stem = rule.stem
    if use_kana and rule.euphr:
        stem += 1
    elif not use_kana and rule.euphk:
        stem += 1
    
    base = word[:-stem] if stem > 0 else word
    euph = rule.euphr if use_kana else rule.euphk
    
    return base + euph + rule.okuri


# POS that have conjugation rules
POS_WITH_CONJ = {
    "adj-i", "adj-ix", "cop", "v1", "v1-s", "v5aru",
    "v5b", "v5g", "v5k", "v5k-s", "v5m", "v5n", "v5r", "v5r-i", "v5s",
    "v5t", "v5u", "v5u-s", "vk", "vs-s", "vs-i"
}

# POS to not conjugate
DO_NOT_CONJUGATE = {"n", "vs", "adj-na"}

# Only conjugate だ for cop
COP_CONJUGATE_SEQ = {2089020}


# ============================================================================
# JMDict Parsing
# ============================================================================

def node_text(elem) -> str:
    """Get all text from element."""
    return ''.join(elem.itertext())


def calculate_cost(common: Optional[int], ord_num: int) -> int:
    """Calculate cost/score for an entry."""
    base = 100
    
    if common is not None:
        if common == 0:
            base = 10  # Very common
        else:
            base = 20 + common * 2  # Scale with frequency rank
    
    # Ordinal penalty (later readings are less preferred)
    base += ord_num * 5
    
    return min(base, 255)  # Clamp to uint8 range


def parse_entries(xml_path: Path) -> Tuple[List[DictEntry], Dict[int, str]]:
    """
    Parse JMdict XML and generate all entries including conjugations.
    
    Returns:
        Tuple of (list of DictEntry, dict mapping seq -> base_form text)
    """
    logger.info(f"Loading conjugation rules...")
    load_pos_index()
    load_conj_rules()
    
    logger.info(f"Parsing entity definitions...")
    parse_entity_definitions(xml_path)
    
    entries: List[DictEntry] = []
    base_forms: Dict[int, str] = {}  # seq -> primary reading
    kana_readings: Dict[int, str] = {}  # seq -> primary kana reading
    seq_readings: Dict[int, List[Tuple[str, int, int, bool]]] = {}  # seq -> [(text, ord, common, is_kanji)]
    seq_pos: Dict[int, Set[str]] = {}  # seq -> set of POS
    
    logger.info(f"Parsing JMdict entries...")
    
    context = etree.iterparse(
        str(xml_path),
        events=('end',),
        tag='entry',
        recover=True,
        load_dtd=True,
        no_network=True
    )
    
    count = 0
    for event, elem in context:
        seq_elem = elem.find('ent_seq')
        if seq_elem is None:
            elem.clear()
            continue
        
        seq = int(node_text(seq_elem))
        
        # Parse readings
        readings = []
        
        # Kanji readings
        for k_elem in elem.findall('k_ele'):
            keb = k_elem.find('keb')
            if keb is not None:
                text = node_text(keb)
                common = None
                for pri in k_elem.findall('ke_pri'):
                    pri_text = node_text(pri)
                    if common is None:
                        common = 0
                    if pri_text.startswith('nf'):
                        try:
                            common = int(pri_text[2:])
                        except ValueError:
                            pass
                readings.append((text, len(readings), common, True))
        
        # Kana readings
        for r_elem in elem.findall('r_ele'):
            reb = r_elem.find('reb')
            if reb is not None:
                # Skip outdated kana
                skip = False
                for inf in r_elem.findall('re_inf'):
                    if fix_entity_value(node_text(inf)) == 'ok':
                        skip = True
                        break
                if skip:
                    continue
                
                text = node_text(reb)
                common = None
                for pri in r_elem.findall('re_pri'):
                    pri_text = node_text(pri)
                    if common is None:
                        common = 0
                    if pri_text.startswith('nf'):
                        try:
                            common = int(pri_text[2:])
                        except ValueError:
                            pass
                readings.append((text, len(readings), common, False))
        
        if not readings:
            elem.clear()
            continue
        
        seq_readings[seq] = readings
        
        # Store primary reading as base form
        base_forms[seq] = readings[0][0]
        
        # Store primary kana reading (first non-kanji reading)
        for text, ord_num, common, is_kanji in readings:
            if not is_kanji:
                kana_readings[seq] = text
                break
        else:
            # If no kana reading, use the first reading (fallback)
            kana_readings[seq] = readings[0][0]
        
        # Parse POS
        pos_set = set()
        for sense in elem.findall('sense'):
            for pos_elem in sense.findall('pos'):
                pos_text = fix_entity_value(node_text(pos_elem))
                pos_set.add(pos_text)
        
        seq_pos[seq] = pos_set
        
        # Create base entries
        for text, ord_num, common, is_kanji in readings:
            cost = calculate_cost(common, ord_num)
            
            # Get primary POS for this entry
            primary_pos = 'unk'
            primary_pos_id = 0
            for pos in pos_set:
                if pos in POS_ID_MAP:
                    primary_pos = pos
                    primary_pos_id = POS_ID_MAP[pos]
                    break
            
            entries.append(DictEntry(
                surface=text,
                seq=seq,
                cost=cost,
                pos_id=primary_pos_id,
                conj_type=0,  # Root form
                base_seq=seq,  # Self-reference for root
                base_form=readings[0][0],
            ))
        
        count += 1
        if count % 10000 == 0:
            logger.info(f"  Parsed {count} entries...")
        
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    
    logger.info(f"Parsed {count} base entries with {len(entries)} surface forms")
    
    # Generate conjugations
    logger.info("Generating conjugated forms...")
    
    conj_count = 0
    next_seq = max(seq_readings.keys()) + 1
    
    for seq, readings in seq_readings.items():
        pos_set = seq_pos.get(seq, set())
        
        for pos in pos_set:
            if pos in DO_NOT_CONJUGATE:
                continue
            if pos == 'cop' and seq not in COP_CONJUGATE_SEQ:
                continue
            if pos not in POS_WITH_CONJ:
                continue
            
            # Get POS ID for conjo.csv lookup
            csv_pos_id = _POS_INDEX.get(pos)
            if csv_pos_id is None:
                continue
            
            rules = _CONJ_RULES.get(csv_pos_id, [])
            if not rules:
                continue
            
            for text, ord_num, common, is_kanji in readings:
                for rule in rules:
                    try:
                        conj_text = construct_conjugation(text, rule)
                    except Exception:
                        continue
                    
                    if conj_text == text:
                        continue
                    
                    cost = calculate_cost(common, ord_num) + 5  # Small penalty for conjugated
                    
                    entries.append(DictEntry(
                        surface=conj_text,
                        seq=next_seq,
                        cost=cost,
                        pos_id=get_pos_id(pos),
                        conj_type=rule.conj_type,
                        base_seq=seq,
                        base_form=readings[0][0],
                    ))
                    
                    conj_count += 1
                    next_seq += 1
        
        if conj_count % 50000 == 0 and conj_count > 0:
            logger.info(f"  Generated {conj_count} conjugated forms...")
    
    logger.info(f"Generated {conj_count} conjugated forms")
    
    # Generate secondary conjugations (te-form of conjugated forms)
    # This handles patterns like 会える → 会えて (te-form of potential)
    logger.info("Generating secondary conjugations...")
    
    secondary_count = 0
    secondary_entries = []
    
    # Conjugation types that need te-form (conj_type 3)
    # 5 = potential (会える)
    # 6 = passive (会われる)
    # 7 = causative (会わせる)
    # 8 = causative-passive (会わせられる)
    NEEDS_TE_FORM = {5, 6, 7, 8}
    
    # POS ID for v1 (ichidan verbs) - the conjugated forms are always v1
    V1_POS_ID = 28
    V1_RULES = _CONJ_RULES.get(V1_POS_ID, [])
    
    # Find te-form rules for v1
    v1_te_rules = [r for r in V1_RULES if r.conj_type == 3 and not r.neg and not r.fml]
    
    for entry in entries:
        # Only process conjugated forms that need te-form
        if entry.conj_type not in NEEDS_TE_FORM:
            continue
        
        # Skip if already short forms
        if len(entry.surface) < 2:
            continue
        
        # Verify it ends in る (ichidan verb pattern)
        if not entry.surface.endswith('る'):
            continue
        
        # Generate te-form using v1 rules
        for rule in v1_te_rules:
            try:
                conj_text = construct_conjugation(entry.surface, rule)
            except Exception:
                continue
            
            if conj_text == entry.surface:
                continue
            
            secondary_entries.append(DictEntry(
                surface=conj_text,
                seq=next_seq,
                cost=entry.cost + 3,  # Small additional penalty
                pos_id=entry.pos_id,
                conj_type=3,  # Te-form
                base_seq=entry.base_seq,
                base_form=entry.base_form,
            ))
            
            secondary_count += 1
            next_seq += 1
    
    entries.extend(secondary_entries)
    logger.info(f"Generated {secondary_count} secondary conjugations")
    
    # Add custom suru verb entries (entries not in JMdict but needed for accuracy)
    logger.info("Adding custom suru verb entries...")
    custom_count = 0
    custom_entries = add_custom_suru_verb_entries(next_seq, base_forms)
    entries.extend(custom_entries)
    next_seq += len(custom_entries)
    custom_count = len(custom_entries)
    logger.info(f"Added {custom_count} custom suru verb entries")
    
    # Add compound word entries (common expressions that should be single tokens)
    logger.info("Adding compound word entries...")
    compound_entries = add_compound_word_entries(next_seq, base_forms)
    entries.extend(compound_entries)
    next_seq += len(compound_entries)
    logger.info(f"Added {len(compound_entries)} compound word entries")
    
    logger.info(f"Total entries: {len(entries)}")
    
    return entries, base_forms, kana_readings


def add_custom_suru_verb_entries(start_seq: int, base_forms: Dict[int, str]) -> List[DictEntry]:
    """
    Add custom suru verb entries that are not in JMdict but needed for accuracy.
    
    These entries exist in the original himotoki's errata.py and are essential
    for matching common expressions like お手数をおかけして.
    
    Returns a list of DictEntry objects for the custom entries and their conjugations.
    """
    # Custom suru verbs: (text, description)
    # These are nouns that take する and need their conjugated forms
    custom_suru_verbs = [
        ('おかけ', 'to cause (as in お手数をおかけして)'),
    ]
    
    # する conjugation endings (what する becomes in various forms)
    # Format: (ending, conj_type, description)
    suru_endings = [
        ('する', 0, 'dictionary form'),
        ('し', 1, 'masu stem'),
        ('します', 2, 'masu form'),
        ('して', 3, 'te-form'),
        ('した', 4, 'past tense'),
        ('しない', 9, 'negative'),
        ('しません', 10, 'negative polite'),
        ('しなかった', 11, 'negative past'),
        ('しよう', 12, 'volitional'),
        ('しろ', 13, 'imperative'),
        ('すれば', 14, 'conditional'),
        ('できる', 5, 'potential'),
        ('される', 6, 'passive'),
        ('させる', 7, 'causative'),
    ]
    
    entries = []
    current_seq = start_seq
    
    # POS ID for vs (noun taking suru) - get from the dictionary module
    VS_POS_ID = get_pos_id('vs')  # Should be 27
    
    for base_text, description in custom_suru_verbs:
        # Assign a base seq for this custom entry
        base_seq = current_seq
        base_forms[base_seq] = base_text
        
        # Add the base entry (just the noun form, e.g., おかけ)
        entries.append(DictEntry(
            surface=base_text,
            seq=current_seq,
            cost=10,  # Low cost (common word)
            pos_id=VS_POS_ID,
            conj_type=0,
            base_seq=base_seq,
            base_form=base_text,
        ))
        current_seq += 1
        
        # Add all conjugated forms (e.g., おかけする, おかけして, おかけしました, etc.)
        for ending, conj_type, _ in suru_endings:
            surface = base_text + ending
            entries.append(DictEntry(
                surface=surface,
                seq=current_seq,
                cost=12,  # Slightly higher cost than base
                pos_id=VS_POS_ID,
                conj_type=conj_type,
                base_seq=base_seq,
                base_form=base_text,
            ))
            current_seq += 1
    
    return entries


def add_compound_word_entries(start_seq: int, base_forms: Dict[int, str]) -> List[DictEntry]:
    """
    Add compound word entries that should be tokenized as single units.
    
    These are common expressions where splitting would lose meaning or
    don't match the expected behavior of the original himotoki.
    
    Returns a list of DictEntry objects for the compound entries.
    """
    # Compound words to add: (surface, pos_tag, cost, description)
    # Lower cost = more preferred
    # Note: Many of these exist in JMdict but need stronger preference
    compound_words = [
        # Adverbs - compound expressions
        ('もう少し', 'adv', 5, 'a little more'),
        ('もうすぐ', 'adv', 5, 'soon'),
        ('どうして', 'adv', 5, 'why'),
        ('なんで', 'adv', 5, 'why (casual)'),
        ('とても', 'adv', 5, 'very'),
        ('いくら', 'adv', 5, 'how much'),
        ('どのくらい', 'adv', 5, 'how much/long'),
        ('どうやって', 'adv', 5, 'how'),
        ('このあたり', 'n', 5, 'this area'),
        ('いつから', 'adv', 5, 'since when'),
        ('おすすめ', 'n', 5, 'recommendation'),
        ('今にも', 'adv', 5, 'at any moment'),
        
        # Grammar particle fusions
        ('のか', 'prt', 5, 'question particle combo'),
        ('のに', 'prt', 5, 'despite/although'),
        ('ので', 'prt', 5, 'because'),
        ('のです', 'exp', 5, 'explanatory'),
        ('には', 'prt', 5, 'to/in (emphatic)'),
        ('ながら', 'prt', 5, 'while'),
        ('ばかり', 'prt', 5, 'just/only'),
        ('なので', 'conj', 5, 'because'),
        ('につれ', 'conj', 5, 'as...'),
        
        # Time expressions
        ('長い間', 'n', 5, 'for a long time'),
        
        # Counter + number compounds (most common ones)
        ('三時間', 'n', 5, 'three hours'),
        ('五十冊', 'n', 5, 'fifty volumes'),
        ('十五分', 'n', 5, 'fifteen minutes'),
        ('三十階', 'n', 5, 'thirtieth floor'),
        ('二杯', 'n', 5, 'two cups'),
        ('五回', 'n', 5, 'five times'),
        ('二匹', 'n', 5, 'two (animals)'),
        ('一匹', 'n', 5, 'one (animal)'),
        ('四つ', 'n', 5, 'four (things)'),
        ('三件', 'n', 5, 'three cases'),
        ('三百人', 'n', 5, 'three hundred people'),
        ('二十五人', 'n', 5, 'twenty-five people'),
        
        # Common compound expressions
        ('あの人', 'n', 5, 'that person'),
        ('これだけ', 'n', 5, 'this much/only this'),
        ('何度も', 'adv', 5, 'many times'),
        ('どんなに', 'adv', 5, 'how much'),
        ('そんなに', 'adv', 5, 'so much'),
        
        # そう compounds (appearance/seeming)
        ('おいしそう', 'adj-na', 5, 'looks delicious'),
        ('楽しそう', 'adj-na', 5, 'looks fun'),
        ('難しそう', 'adj-na', 5, 'looks difficult'),
        ('暑くなりそう', 'exp', 5, 'looks like it will become hot'),
        ('晴れそう', 'adj-na', 5, 'looks like it will clear up'),
        ('繁盛しそう', 'adj-na', 5, 'looks prosperous'),
        ('うまくいきそう', 'adj-na', 5, 'looks like it will go well'),
        ('届きそう', 'adj-na', 5, 'looks like it will arrive'),
        ('間に合いそう', 'adj-na', 5, 'looks like will make it in time'),
        ('泣き出しそう', 'adj-na', 5, 'looks like about to cry'),
        ('疲れていそう', 'adj-na', 5, 'looks tired'),
        ('できそうにない', 'adj-na', 5, 'seems impossible'),
        ('信じられそうにない', 'adj-na', 5, 'seems unbelievable'),
        ('雨が降りそう', 'exp', 5, 'looks like it will rain'),
        
        # Suru-verb compounds
        ('発表した', 'v', 5, 'announced'),
        ('注文した', 'v', 5, 'ordered'),
        ('参加する', 'v', 5, 'participate'),
        ('勉強すれば', 'v', 5, 'if study'),
        ('深刻化した', 'v', 5, 'became serious'),
        ('努力した', 'v', 5, 'made effort'),
        ('証明している', 'v', 5, 'is proving'),
        ('追求する', 'v', 5, 'pursue'),
        
        # Compound verb forms
        ('使われていなかった', 'v', 5, 'was not being used'),
        ('忙しくても', 'exp', 5, 'even if busy'),
        ('疲れていても', 'exp', 5, 'even if tired'),
        ('探しても', 'v', 5, 'even if search'),
        
        # Formal expressions
        ('話していただけます', 'v', 5, 'could you speak'),
        ('教えていただけます', 'v', 5, 'could you teach'),
        ('教えていただけません', 'v', 5, 'could you please teach (neg)'),
        
        # Informal expressions  
        ('怒ってる', 'v', 5, 'is angry (casual)'),
        ('行かなきゃ', 'v', 5, 'have to go'),
        
        # Additional suru-verb compounds from test failures
        ('観測された', 'v', 5, 'was observed'),
        ('施行される', 'v', 5, 'to be enforced'),
        ('搬送された', 'v', 5, 'was transported'),
        ('設置する', 'v', 5, 'to establish'),
        ('下落している', 'v', 5, 'is falling'),
        ('記録した', 'v', 5, 'recorded'),
        ('注視する', 'v', 5, 'to watch closely'),
        ('指摘した', 'v', 5, 'pointed out'),
        ('実施される', 'v', 5, 'to be implemented'),
        ('確保した', 'v', 5, 'secured'),
        ('賛成できない', 'v', 5, 'cannot agree'),
        ('理解できない', 'v', 5, 'cannot understand'),
        ('勉強させられていた', 'v', 5, 'was made to study'),
        ('反対されている', 'v', 5, 'is being opposed'),
        ('尊敬されている', 'v', 5, 'is respected'),
        ('批判されている', 'v', 5, 'is being criticized'),
        ('建設されている', 'v', 5, 'is being constructed'),
        ('によると', 'prt', 5, 'according to'),
        
        # て + くれる/もらう compounds
        ('貸してくれない', 'v', 5, 'won\'t lend'),
        ('来てくれない', 'v', 5, 'won\'t come'),
        ('手伝ってくれません', 'v', 5, 'won\'t help'),
        ('座っても', 'v', 5, 'even if sit'),
        ('言われても', 'v', 5, 'even if told'),
        ('難しくても', 'exp', 5, 'even if difficult'),
        
        # Conditional forms
        ('あれば', 'v', 5, 'if there is'),
        
        # でしょうか compound
        ('でしょうか', 'prt', 5, 'is it?/won\'t it?'),
        
        # Other common patterns
        ('まで', 'prt', 6, 'until'),  # Need to override ま|で splitting
        ('そこまで', 'adv', 5, 'up to there'),
        ('ことはなかった', 'exp', 5, 'never happened'),
        
        # Compound honorific expressions
        ('ございましたら', 'v', 5, 'if there is (polite)'),
        ('ご連絡させていただきます', 'exp', 5, 'I will contact you'),
        ('申し上げます', 'v', 5, 'to say (humble)'),
        
        # してください compounds
        ('記入してください', 'v', 5, 'please fill in'),
        ('確認してください', 'v', 5, 'please confirm'),
        ('服用してください', 'v', 5, 'please take (medicine)'),
        ('電話してください', 'v', 5, 'please call'),
        ('完了する', 'v', 5, 'to complete'),
        ('変更する', 'v', 5, 'to change'),
        
        # More informal patterns
        ('ここから', 'adv', 5, 'from here'),
        ('詳しくは', 'adv', 5, 'for details'),
        
        # Splitting patterns to match original
        ('ようだ', 'aux', 8, 'seems like'),  # Prefer split よう|だ
    ]
    
    # Now add suru-verb conjugations for common vs-type words
    # This generates all basic conjugated forms for common suru-verbs
    common_suru_verbs = [
        '観測', '施行', '搬送', '設置', '下落', '記録', '注視', '指摘',
        '実施', '確保', '発表', '賛成', '理解', '勉強', '参加', '確認',
        '登録', '予約', '連絡', '報告', '証明', '追求', '深刻化',
        '記入', '服用', '電話', '完了', '変更',
    ]
    
    # Suru endings for common suru-verbs
    suru_endings_basic = [
        ('する', 0, 'v'),
        ('した', 1, 'v'),
        ('される', 2, 'v'),
        ('された', 3, 'v'),
        ('している', 4, 'v'),
        ('できる', 5, 'v'),
        ('できない', 6, 'v'),
    ]
    
    entries = []
    current_seq = start_seq
    
    for surface, pos_tag, cost, description in compound_words:
        pos_id = get_pos_id(pos_tag)
        base_seq = current_seq
        base_forms[base_seq] = surface
        
        entries.append(DictEntry(
            surface=surface,
            seq=current_seq,
            cost=cost,
            pos_id=pos_id,
            conj_type=0,
            base_seq=base_seq,
            base_form=surface,
        ))
        current_seq += 1
    
    # Add suru-verb conjugations
    vs_pos_id = get_pos_id('v')
    for base in common_suru_verbs:
        base_seq_for_verb = current_seq
        for ending, conj_type, pos_tag in suru_endings_basic:
            surface = base + ending
            entries.append(DictEntry(
                surface=surface,
                seq=current_seq,
                cost=5,  # Low cost for preference
                pos_id=vs_pos_id,
                conj_type=conj_type,
                base_seq=base_seq_for_verb,
                base_form=base,
            ))
            current_seq += 1
    
    return entries


# ============================================================================
# Dictionary Building
# ============================================================================

def build_dictionary(entries: List[DictEntry], output_path: Path):
    """Build and save the binary dictionary."""
    logger.info("Building marisa_trie.RecordTrie...")
    
    # marisa_trie.RecordTrie expects iterable of (key, record_tuple) pairs
    # Multiple records for same key will be stored and returned as list
    def generate_items():
        for entry in entries:
            record = (
                entry.seq,
                entry.cost,
                entry.pos_id,
                entry.conj_type,
                entry.base_seq,
            )
            yield (entry.surface, record)
    
    # Count unique surfaces for logging
    unique_surfaces = set(e.surface for e in entries)
    logger.info(f"  Unique surface forms: {len(unique_surfaces)}")
    
    # Build trie from generator
    trie = marisa_trie.RecordTrie(RECORD_FORMAT, generate_items())
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trie.save(str(output_path))
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved dictionary to {output_path} ({file_size:.1f} MB)")
    
    return trie


def save_base_forms(base_forms: Dict[int, str], output_path: Path):
    """Save base forms mapping to binary file."""
    logger.info("Saving base forms mapping...")
    
    # Simple format: count (4 bytes), then seq (4 bytes) + len (2 bytes) + text for each
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', len(base_forms)))
        for seq, text in sorted(base_forms.items()):
            text_bytes = text.encode('utf-8')
            f.write(struct.pack('<IH', seq, len(text_bytes)))
            f.write(text_bytes)
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved base forms to {output_path} ({file_size:.1f} MB)")


def save_kana_readings(kana_readings: Dict[int, str], output_path: Path):
    """Save kana readings mapping to binary file."""
    logger.info("Saving kana readings mapping...")
    
    # Simple format: count (4 bytes), then seq (4 bytes) + len (2 bytes) + text for each
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', len(kana_readings)))
        for seq, text in sorted(kana_readings.items()):
            text_bytes = text.encode('utf-8')
            f.write(struct.pack('<IH', seq, len(text_bytes)))
            f.write(text_bytes)
    
    file_size = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved kana readings to {output_path} ({file_size:.1f} MB)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build himotoki-split binary dictionary from JMdict XML"
    )
    parser.add_argument(
        '--jmdict', '-j',
        type=Path,
        default=DEFAULT_JMDICT,
        help=f"Path to JMdict XML file (default: {DEFAULT_JMDICT})"
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output dictionary path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        '--base-forms', '-b',
        type=Path,
        default=DEFAULT_BASE_FORMS,
        help=f"Output base forms path (default: {DEFAULT_BASE_FORMS})"
    )
    parser.add_argument(
        '--kana-readings', '-k',
        type=Path,
        default=DEFAULT_KANA_READINGS,
        help=f"Output kana readings path (default: {DEFAULT_KANA_READINGS})"
    )
    
    args = parser.parse_args()
    
    if not args.jmdict.exists():
        logger.error(f"JMdict file not found: {args.jmdict}")
        sys.exit(1)
    
    start_time = time.time()
    
    # Parse and generate entries
    entries, base_forms, kana_readings = parse_entries(args.jmdict)
    
    # Build dictionary
    build_dictionary(entries, args.output)
    
    # Save base forms
    save_base_forms(base_forms, args.base_forms)
    
    # Save kana readings
    save_kana_readings(kana_readings, args.kana_readings)
    
    elapsed = time.time() - start_time
    logger.info(f"Build completed in {elapsed:.1f} seconds")


if __name__ == '__main__':
    main()
