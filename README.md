# himotoki-split

<div align="center">

**Lightweight Japanese Morphological Analyzer**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Package Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](https://pypi.org/project/himotoki-split/)

*A pure tokenizer inspired by [SudachiPy](https://github.com/WorksApplications/SudachiPy). No heavy database required.*

</div>

---

## 🚀 Overview

`himotoki-split` is a high-performance, DB-less Japanese tokenizer designed for environments where memory and startup time are critical. While the original `himotoki` relies on a 3GB SQLite database for full linguistic analysis, `himotoki-split` uses a compact **27MB binary dictionary** powered by `marisa-trie`, allowing for near-instant execution.

### ✨ Key Features

- 🏎️ **Instant Startup**: Dictionary loads in `<1ms` using memory-mapped files.
- 📦 **Compact Footprint**: Only **27MB** (compared to 3GB for the full database version).
- 🧩 **Pure Python Core**: Minimal dependencies (`marisa-trie`, `lxml`).
- 💎 **JMdict-based**: Leverages industry-standard Japanese dictionary data.
- 🔗 **Deep Integration**: Seamlessly compatible with the full `himotoki` library via `base_form_id`.

---

## 📦 Installation

```bash
pip install himotoki-split
```

---

## 📺 Demo

You can try the tokenizer directly from your terminal:

```bash
# Basic tokenization
himotoki-split "俺の力を見せてやる"
# Output: 俺 | の | 力 | を | 見せて | やる

# Detailed analysis (multi-candidate)
himotoki-split --json "絶対に負けない" | jq
```

---

## 📝 Usage Examples

### 1. Basic Tokenization
Split text into morphemes with surface form, base form, and part-of-speech (POS) tags.

```python
import himotoki_split

text = "今日は天気がいいですね。"
tokens = himotoki_split.tokenize(text)

print(f"{'Surface':<10} | {'Base Form':<10} | {'POS':<10}")
print("-" * 35)
for token in tokens:
    print(f"{token.surface:<10} | {token.base_form:<10} | {token.pos:<10}")

# Output:
# Surface    | Base Form  | POS       
# -----------------------------------
# 今日         | 今日         | unk       
# は          | は          | unk       
# 天気         | 天気         | n         
# が          | が          | conj      
# いい         | いい         | adj-ix    
# です         | です         | cop       
# ね          | ね          | int       
# 。          | 。          | punc
```

### 2. Multi-Candidate Analysis
Gain insight into scoring and Alternative segmentations.

```python
results = himotoki_split.analyze("今日は", limit=3)

for tokens, score in results:
    surfaces = [t.surface for t in tokens]
    print(f"[{score:.4f}] {' + '.join(surfaces)}")
```

---

## 🏗️ Architecture

`himotoki-split` achieves its lightness by separating the **segmentation engine** from the **lexical database**.

1. **Dictionary Construction**: JMdict entries are compiled into a double-array trie (`marisa-trie`).
2. **Cost-based Segmentation**: Uses a Viterbi-like algorithm with pre-calculated connection costs to find the most probable path.
3. **Lazy Loading**: The dictionary is memory-mapped, ensuring that only the necessary parts are loaded into RAM.

---

## 📊 Comparison

| Feature | himotoki-split | himotoki (Full) | SudachiPy (Small) |
| :--- | :---: | :---: | :---: |
| **Dictionary Size** | ~27 MB | ~3 GB | ~10 MB |
| **Database Required** | No | Yes (SQLite) | No |
| **Startup Time** | <1ms | ~500ms | ~50ms |
| **Definitions**| IDs Only | Full Definitions | None |

---

## 🛠️ CLI Usage

Standard text output:
```bash
himotoki-split "俺の力を見せてやる"
```

JSON output for integrations:
```bash
himotoki-split --json "絶対に負けない"
```

---

## 🤝 Related Projects

- [himotoki](https://github.com/msr2903/himotoki) - The full Japanese analyzer featuring complete dictionary definitions.
- [ichiran](https://github.com/tshatrov/ichiran) - The original Lisp implementation and inspiration for the segmentation logic.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
