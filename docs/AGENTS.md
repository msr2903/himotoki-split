# Himotoki-Split AI Agent Context Guide

A concise technical reference for AI agents working with the himotoki-split codebase.

---

## Project Overview

**himotoki-split** is a lightweight Japanese morphological analyzer - a pure tokenizer like SudachiPy with no database required. Uses a compact binary dictionary (~27MB) for instant startup.

| Aspect | Details |
|--------|---------|
| **Language** | Python 3.10+ |
| **Dictionary** | marisa-trie binary (~27MB) |
| **Data Source** | JMdict (EDRDG) |
| **Algorithm** | Viterbi-style dynamic programming |
| **Package Manager** | pip/uv with pyproject.toml |

### Project Structure

```
himotoki-split/
├── himotoki_split/              # Main package
│   ├── __init__.py              # Public API: tokenize(), analyze(), warm_up()
│   ├── tokenizer.py             # Core tokenization algorithm (Viterbi DP)
│   ├── dictionary.py            # Binary dictionary loading and lookup
│   ├── characters.py            # Character utilities (kana/kanji detection)
│   ├── constants.py             # SEQ numbers, conjugation IDs
│   ├── cli.py                   # Command-line interface
│   └── data/                    # Binary dictionary files
│       └── himotoki.dic         # marisa-trie dictionary
├── scripts/                     # Developer utilities
│   ├── build_dictionary.py      # Build dictionary from JMdict
│   ├── compare_accuracy.py      # Compare with original himotoki
│   └── benchmark.py             # Performance benchmarks
├── tests/                       # Test suite
├── data/                        # Source data (JMdict XML, conjugation CSVs)
└── pyproject.toml               # Project configuration
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     PUBLIC API (__init__.py)                     │
│  tokenize(), analyze(), warm_up(), get_version()                │
│  Token: surface, reading, pos, base_form, base_form_id          │
└────────────────────────────────────────────────────────────────┬┘
                                                                  │
┌────────────────────────────────────────────────────────────────▼┐
│                  TOKENIZER (tokenizer.py)                       │
│  - find_all_matches(): find all word candidates                 │
│  - find_best_path(): Viterbi DP to find optimal segmentation    │
│  - calculate_segment_score(): scoring with length preference    │
└────────────────────────────────────────────────────────────────┬┘
                                                                  │
┌────────────────────────────────────────────────────────────────▼┐
│                  DICTIONARY (dictionary.py)                     │
│  - marisa_trie.RecordTrie for O(1) prefix lookup                │
│  - Memory-mapped: instant loading (<1ms)                        │
│  - WordEntry: surface, seq, cost, pos_id, conj_type, base_seq   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Data Structures

### Token (Public API)

```python
@dataclass
class Token:
    surface: str       # As appears in text (e.g., "食べた")
    reading: str       # Reading in hiragana
    pos: str           # Part of speech (e.g., "v1", "n", "prt")
    base_form: str     # Dictionary form (e.g., "食べる")
    base_form_id: int  # JMdict sequence ID
    start: int         # Start position in text
    end: int           # End position in text
```

### WordEntry (Internal)

```python
@dataclass
class WordEntry:
    surface: str       # Surface form text
    seq: int           # JMdict sequence ID
    cost: int          # Pre-calculated cost for Viterbi
    pos_id: int        # Compact POS ID (1 byte)
    conj_type: int     # Conjugation type (0 = root form)
    base_seq: int      # Base form sequence ID
```

### Binary Record Format

Each dictionary entry is 12 bytes:
- `seq`: int32 (4 bytes) - JMdict sequence ID
- `cost`: int16 (2 bytes) - Viterbi cost
- `pos_id`: uint8 (1 byte) - Part of speech
- `conj_type`: uint8 (1 byte) - Conjugation type
- `base_seq`: int32 (4 bytes) - Base form sequence

---

## Scoring Algorithm

The scoring prioritizes **longer words** to prevent over-splitting:

```python
def calculate_segment_score(surface: str, entry: WordEntry) -> float:
    # Length is PRIMARY factor (50 points per character)
    length_bonus = len(surface) * 50
    
    # Cost penalty (lower cost = higher score)
    cost_penalty = min(entry.cost, 100)
    
    base_score = length_bonus - cost_penalty
    
    # Penalize words that suspiciously end with particles
    # (e.g., prefer 今日|は over 今日は)
    if ends_with_particle(surface):
        base_score -= 60
    
    return base_score
```

---

## Dictionary Building

Build dictionary from JMdict XML:

```bash
python scripts/build_dictionary.py
```

This creates:
- `himotoki_split/data/himotoki.dic` - marisa-trie binary (~27MB)

The build process:
1. Parses JMdict XML
2. Extracts surface forms, readings, POS, sequence IDs
3. Generates conjugated forms
4. Calculates costs based on frequency/commonness
5. Writes to marisa-trie format

---

## Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Run accuracy comparison
python scripts/compare_accuracy.py

# Benchmark performance
python scripts/benchmark.py

# Basic tokenization
python -c "import himotoki_split; print(himotoki_split.tokenize('今日は天気がいい'))"
```

---

## POS Tag Reference

| ID | Tag | Description |
|----|-----|-------------|
| 1-5 | n, n-adv, n-pref, n-suf, n-t | Nouns |
| 10-30 | v1, v5*, vk, vs | Verbs |
| 40-46 | adj-i, adj-na, adj-no | Adjectives |
| 50-51 | adv, adv-to | Adverbs |
| 60-62 | aux, aux-v, aux-adj | Auxiliaries |
| 70-74 | conj, cop, ctr, exp, int | Other |
| 80-84 | pn, pref, prt, suf, unc | Particles, etc. |

---

## Related Projects

- [himotoki](https://github.com/msr2903/himotoki) - Full Japanese analyzer with SQLite dictionary (3GB)
- [ichiran](https://github.com/tshatrov/ichiran) - Original Lisp implementation
- [SudachiPy](https://github.com/WorksApplications/SudachiPy) - Another Japanese tokenizer
