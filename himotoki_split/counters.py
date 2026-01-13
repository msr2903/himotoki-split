"""
Counter expression recognition module for himotoki-split.
Ported from original himotoki's counters.py.

Counter expressions combine a number with a counter suffix to form compound words.
Examples:
- 三匹 (sanbiki) = 三 (san, three) + 匹 (hiki, counter for small animals)
- 五冊 (gosatsu) = 五 (go, five) + 冊 (satsu, counter for books)
- 千人 (sennin) = 千 (sen, thousand) + 人 (nin, counter for people)

This module provides counter recognition without database access.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Set


# ============================================================================
# Japanese Number Parsing
# ============================================================================

# Number kanji to value mapping
KANJI_NUMBERS: Dict[str, int] = {
    '零': 0, '〇': 0,
    '一': 1, '壱': 1,
    '二': 2, '弐': 2,
    '三': 3, '参': 3,
    '四': 4,
    '五': 5,
    '六': 6,
    '七': 7,
    '八': 8,
    '九': 9,
    '十': 10,
    '百': 100,
    '千': 1000,
    '万': 10000,
    '億': 100000000,
}

# Arabic digit to value (both half-width and full-width)
DIGIT_VALUES: Dict[str, int] = {
    '0': 0, '０': 0,
    '1': 1, '１': 1,
    '2': 2, '２': 2,
    '3': 3, '３': 3,
    '4': 4, '４': 4,
    '5': 5, '５': 5,
    '6': 6, '６': 6,
    '7': 7, '７': 7,
    '8': 8, '８': 8,
    '9': 9, '９': 9,
}

# Digit to kana reading
DIGIT_TO_KANA: Dict[int, str] = {
    0: 'ゼロ',
    1: 'いち',
    2: 'に',
    3: 'さん',
    4: 'よん',  # Common reading (し is also valid in some contexts)
    5: 'ご',
    6: 'ろく',
    7: 'なな',  # Common reading (しち is also valid)
    8: 'はち',
    9: 'きゅう',  # Common reading (く is also valid)
}

# Power values to kana
POWER_TO_KANA: Dict[int, str] = {
    1: 'じゅう',  # 10
    2: 'ひゃく',  # 100
    3: 'せん',    # 1000
    4: 'まん',    # 10000
}


def parse_number(text: str) -> Optional[int]:
    """
    Parse a Japanese number string into an integer.
    
    Supports:
    - Arabic numerals (1, 2, 3 or １, ２, ３)
    - Kanji numerals (一, 二, 三)
    - Mixed (十二, 百三十四)
    
    Returns None if the text is not a valid number.
    """
    if not text:
        return None
    
    # Try Arabic numerals first
    arabic_str = ''
    for char in text:
        if char in DIGIT_VALUES:
            arabic_str += str(DIGIT_VALUES[char])
        elif char.isdigit():
            arabic_str += char
        else:
            break
    
    if arabic_str and len(arabic_str) == len(text):
        return int(arabic_str)
    
    # Try kanji numerals
    return parse_kanji_number(text)


def parse_kanji_number(text: str) -> Optional[int]:
    """Parse a kanji number string into an integer."""
    if not text:
        return None
    
    result = 0
    current = 0
    
    for char in text:
        if char not in KANJI_NUMBERS:
            return None
        
        value = KANJI_NUMBERS[char]
        
        if value >= 10:
            # Multiplier (十, 百, 千, 万, 億)
            if current == 0:
                current = 1
            
            if value >= 10000:
                # 万 or higher - adds to result
                result += current * value
                current = 0
            else:
                # 十, 百, 千 - multiplies current
                current *= value
                result += current
                current = 0
        else:
            # Digit
            if current > 0:
                result += current
            current = value
    
    # Add any remaining current
    result += current
    
    return result if result > 0 or text in ('零', '〇') else None


def number_to_kana(n: int) -> str:
    """Convert a number to its kana reading."""
    if n == 0:
        return 'ゼロ'
    
    parts = []
    
    # Handle 10000s
    if n >= 10000:
        man = n // 10000
        if man > 1:
            parts.append(number_to_kana(man))
        parts.append('まん')
        n %= 10000
    
    # Handle 1000s
    if n >= 1000:
        sen = n // 1000
        if sen == 3:
            parts.append('さんぜん')  # 3000 = sanzen (rendaku)
        elif sen == 8:
            parts.append('はっせん')  # 8000 = hassen (gemination)
        elif sen > 1:
            parts.append(DIGIT_TO_KANA.get(sen, ''))
            parts.append('せん')
        else:
            parts.append('せん')
        n %= 1000
    
    # Handle 100s
    if n >= 100:
        hyaku = n // 100
        if hyaku == 3:
            parts.append('さんびゃく')  # 300 = sanbyaku
        elif hyaku == 6:
            parts.append('ろっぴゃく')  # 600 = roppyaku
        elif hyaku == 8:
            parts.append('はっぴゃく')  # 800 = happyaku
        elif hyaku > 1:
            parts.append(DIGIT_TO_KANA.get(hyaku, ''))
            parts.append('ひゃく')
        else:
            parts.append('ひゃく')
        n %= 100
    
    # Handle 10s
    if n >= 10:
        juu = n // 10
        if juu > 1:
            parts.append(DIGIT_TO_KANA.get(juu, ''))
        parts.append('じゅう')
        n %= 10
    
    # Handle ones
    if n > 0:
        parts.append(DIGIT_TO_KANA.get(n, ''))
    
    return ''.join(parts)


# ============================================================================
# Counter Data (Pre-computed from JMdict)
# ============================================================================

@dataclass(slots=True)
class CounterInfo:
    """Information about a counter."""
    text: str       # Counter text (kanji or kana)
    kana: str       # Reading in kana
    seq: int        # JMdict sequence ID
    common: bool    # Whether this is a common counter


# Common counters with their readings and special phonetic rules
# Format: (counter_text, kana_reading, seq, digit_opts)
# digit_opts: dict mapping digit to phonetic modifications
# 'g' = gemination, 'r' = rendaku, 'h' = handakuten
COMMON_COUNTERS: Dict[str, Tuple[str, int, Optional[Dict]]] = {
    # Small animals counter - 匹
    '匹': ('ひき', 1424040, {
        1: ['g', 'h'],   # 一匹 = いっぴき
        3: ['r'],        # 三匹 = さんびき
        6: ['g', 'h'],   # 六匹 = ろっぴき
        8: ['g', 'h'],   # 八匹 = はっぴき
        10: ['g', 'h'],  # 十匹 = じゅっぴき
        100: ['g', 'h'], # 百匹 = ひゃっぴき
    }),
    # Books/volumes counter - 冊
    '冊': ('さつ', 1320180, {
        1: ['g'],        # 一冊 = いっさつ
        8: ['g'],        # 八冊 = はっさつ
        10: ['g'],       # 十冊 = じゅっさつ
    }),
    # Cups/glasses counter - 杯
    '杯': ('はい', 1423860, {
        1: ['g', 'h'],   # 一杯 = いっぱい
        3: ['r'],        # 三杯 = さんばい
        6: ['g', 'h'],   # 六杯 = ろっぱい
        8: ['g', 'h'],   # 八杯 = はっぱい
        10: ['g', 'h'],  # 十杯 = じゅっぱい
    }),
    # Long objects counter - 本
    '本': ('ほん', 1211960, {
        1: ['g', 'h'],   # 一本 = いっぽん
        3: ['r'],        # 三本 = さんぼん
        6: ['g', 'h'],   # 六本 = ろっぽん
        8: ['g', 'h'],   # 八本 = はっぽん
        10: ['g', 'h'],  # 十本 = じゅっぽん
    }),
    # Floors counter - 階
    '階': ('かい', 1204100, {
        3: ['g'],        # 三階 = さんがい (rendaku)
    }),
    # Buildings counter - 軒
    '軒': ('けん', 1247710, {
        3: ['g'],        # 三軒 = さんげん (rendaku)
    }),
    # Times counter - 回
    '回': ('かい', 1204040, None),
    # Years counter - 年
    '年': ('ねん', 1194480, {
        4: 'よ',         # 四年 = よねん (not よんねん)
    }),
    # Months counter - 月
    '月': ('がつ', 1255430, {
        4: 'し',         # 四月 = しがつ (not よんがつ)
        7: 'しち',       # 七月 = しちがつ (not なながつ)
        9: 'く',         # 九月 = くがつ (not きゅうがつ)
    }),
    # Days counter - 日
    '日': ('にち', 2083100, None),  # 日 has special readings handled separately
    # People counter - 人
    '人': ('にん', 2149890, None),  # 人 has special readings for 1-2
    # Times/degrees counter - 度
    '度': ('ど', 1543410, None),
    # Machines/vehicles counter - 台
    '台': ('だい', 1412350, None),
    # Places/rank counter - 位
    '位': ('い', 1166410, None),
    # Sheets counter - 枚
    '枚': ('まい', 1529580, None),
    # Individual items counter - 個
    '個': ('こ', 1253170, {
        1: ['g'],        # 一個 = いっこ
        6: ['g'],        # 六個 = ろっこ
        8: ['g'],        # 八個 = はっこ
        10: ['g'],       # 十個 = じゅっこ
    }),
    # General counter - つ
    'つ': ('つ', 2136920, None),  # Special readings for 1-9
    # Number counter (号) - rooms, issues, etc.
    '号': ('ごう', 1260840, None),
    # Weeks counter
    '週': ('しゅう', 1333450, {
        1: ['g'],        # 一週 = いっしゅう
    }),
    # Seconds counter
    '秒': ('びょう', 1482960, None),
    # Minutes counter
    '分': ('ふん', 1582970, {
        1: ['g', 'h'],   # 一分 = いっぷん
        3: ['h'],        # 三分 = さんぷん
        4: 'よん',       # 四分 = よんぷん
        6: ['g', 'h'],   # 六分 = ろっぷん
        8: ['g', 'h'],   # 八分 = はっぷん
        10: ['g', 'h'],  # 十分 = じゅっぷん
    }),
    # Hours counter
    '時': ('じ', 1319440, {
        4: 'よ',         # 四時 = よじ
        7: 'しち',       # 七時 = しちじ
        9: 'く',         # 九時 = くじ
    }),
    # Yen counter
    '円': ('えん', 1171810, None),
    # Pages counter
    'ページ': ('ぺーじ', 1094420, None),
    # Kilograms
    'キロ': ('きろ', 1063420, None),
    # Grams
    'グラム': ('ぐらむ', 1063530, None),
    # Centimeters
    'センチ': ('せんち', 1073330, None),
    # Meters
    'メートル': ('めーとる', 1095200, None),
}

# Days of the month with kun readings (1-10, 14, 20, 24, 30)
DAYS_KUN_READINGS: Dict[int, str] = {
    1: 'ついたち',
    2: 'ふつか',
    3: 'みっか',
    4: 'よっか',
    5: 'いつか',
    6: 'むいか',
    7: 'なのか',
    8: 'ようか',
    9: 'ここのか',
    10: 'とおか',
    14: 'じゅうよっか',
    20: 'はつか',
    24: 'にじゅうよっか',
    30: 'みそか',
}

# People counter special readings
PEOPLE_KUN_READINGS: Dict[int, str] = {
    1: 'ひとり',
    2: 'ふたり',
}

# つ counter special readings (1-9)
TSU_READINGS: Dict[int, str] = {
    1: 'ひとつ',
    2: 'ふたつ',
    3: 'みっつ',
    4: 'よっつ',
    5: 'いつつ',
    6: 'むっつ',
    7: 'ななつ',
    8: 'やっつ',
    9: 'ここのつ',
    10: 'とお',
}


# ============================================================================
# Phonetic Rules
# ============================================================================

def geminate(kana: str) -> str:
    """
    Apply gemination (sokuon) to the end of a kana string.
    E.g., "いち" -> "いっ"
    """
    if not kana:
        return kana
    return kana[:-1] + 'っ'


def rendaku(kana: str, handakuten: bool = False) -> str:
    """
    Apply rendaku (voicing) or handakuten to the beginning of a kana string.
    E.g., "ひき" -> "びき" (rendaku) or "ぴき" (handakuten)
    """
    if not kana:
        return kana
    
    first_char = kana[0]
    rest = kana[1:]
    
    if handakuten:
        # h -> p
        h_to_p = {
            'は': 'ぱ', 'ひ': 'ぴ', 'ふ': 'ぷ', 'へ': 'ぺ', 'ほ': 'ぽ',
            'ハ': 'パ', 'ヒ': 'ピ', 'フ': 'プ', 'ヘ': 'ペ', 'ホ': 'ポ',
        }
        if first_char in h_to_p:
            return h_to_p[first_char] + rest
    else:
        # Voicing: k->g, s->z, t->d, h->b
        voicing = {
            'か': 'が', 'き': 'ぎ', 'く': 'ぐ', 'け': 'げ', 'こ': 'ご',
            'さ': 'ざ', 'し': 'じ', 'す': 'ず', 'せ': 'ぜ', 'そ': 'ぞ',
            'た': 'だ', 'ち': 'ぢ', 'つ': 'づ', 'て': 'で', 'と': 'ど',
            'は': 'ば', 'ひ': 'び', 'ふ': 'ぶ', 'へ': 'べ', 'ほ': 'ぼ',
            'カ': 'ガ', 'キ': 'ギ', 'ク': 'グ', 'ケ': 'ゲ', 'コ': 'ゴ',
            'サ': 'ザ', 'シ': 'ジ', 'ス': 'ズ', 'セ': 'ゼ', 'ソ': 'ゾ',
            'タ': 'ダ', 'チ': 'ヂ', 'ツ': 'ヅ', 'テ': 'デ', 'ト': 'ド',
            'ハ': 'バ', 'ヒ': 'ビ', 'フ': 'ブ', 'ヘ': 'ベ', 'ホ': 'ボ',
        }
        if first_char in voicing:
            return voicing[first_char] + rest
    
    return kana


# ============================================================================
# Counter Expression Recognition
# ============================================================================

@dataclass(slots=True)
class CounterExpression:
    """A recognized counter expression."""
    text: str           # Full text (e.g., "三匹")
    reading: str        # Reading in kana (e.g., "さんびき")
    number: int         # Numeric value
    counter: str        # Counter part (e.g., "匹")
    counter_seq: int    # JMdict seq for the counter
    start: int          # Start position in original text
    end: int            # End position in original text


def find_counter_expression(
    text: str,
    start: int = 0,
) -> Optional[CounterExpression]:
    """
    Try to find a counter expression starting at the given position.
    
    Returns:
        CounterExpression if found, None otherwise
    """
    if start >= len(text):
        return None
    
    # Find the number part
    num_end = start
    while num_end < len(text):
        char = text[num_end]
        if char in KANJI_NUMBERS or char in DIGIT_VALUES or char.isdigit():
            num_end += 1
        else:
            break
    
    if num_end == start:
        return None
    
    number_text = text[start:num_end]
    number_value = parse_number(number_text)
    if number_value is None:
        return None
    
    # Try to find a matching counter
    remaining = text[num_end:]
    
    for counter_text, (counter_kana, counter_seq, digit_opts) in COMMON_COUNTERS.items():
        if remaining.startswith(counter_text):
            end = num_end + len(counter_text)
            reading = generate_counter_reading(number_value, number_text, counter_kana, digit_opts)
            
            # Special handling for 日 (days)
            if counter_text == '日' and number_value in DAYS_KUN_READINGS:
                reading = DAYS_KUN_READINGS[number_value]
            
            # Special handling for 人 (people)
            if counter_text == '人' and number_value in PEOPLE_KUN_READINGS:
                reading = PEOPLE_KUN_READINGS[number_value]
            
            # Special handling for つ counter
            if counter_text == 'つ' and number_value in TSU_READINGS:
                reading = TSU_READINGS[number_value]
            
            return CounterExpression(
                text=text[start:end],
                reading=reading,
                number=number_value,
                counter=counter_text,
                counter_seq=counter_seq,
                start=start,
                end=end,
            )
    
    return None


def generate_counter_reading(
    number: int,
    number_text: str,
    counter_kana: str,
    digit_opts: Optional[Dict],
) -> str:
    """Generate the reading for a counter expression."""
    number_kana = number_to_kana(number)
    
    if digit_opts is None:
        return number_kana + counter_kana
    
    # Get the relevant digit for phonetic rules
    digit = number % 10
    if digit == 0:
        for power in [10, 100, 1000, 10000]:
            if number % power == 0 and number % (power * 10) != 0:
                digit = power
                break
    
    if digit not in digit_opts:
        return number_kana + counter_kana
    
    opts = digit_opts[digit]
    
    if isinstance(opts, str):
        # Replace the digit reading
        # Get what we need to replace
        digit_kana = DIGIT_TO_KANA.get(digit, '')
        if digit_kana and number_kana.endswith(digit_kana):
            number_kana = number_kana[:-len(digit_kana)] + opts
        else:
            number_kana = opts
        return number_kana + counter_kana
    
    # Apply modifications
    result_number = number_kana
    result_counter = counter_kana
    
    for opt in opts:
        if opt == 'g':
            result_number = geminate(result_number)
        elif opt == 'r':
            result_counter = rendaku(result_counter)
        elif opt == 'h':
            result_counter = rendaku(result_counter, handakuten=True)
    
    return result_number + result_counter


def find_all_counters(text: str) -> List[CounterExpression]:
    """
    Find all counter expressions in a text.
    
    Returns:
        List of CounterExpression objects
    """
    results = []
    i = 0
    
    while i < len(text):
        counter = find_counter_expression(text, i)
        if counter:
            results.append(counter)
            i = counter.end
        else:
            i += 1
    
    return results


def is_counter_start(text: str, pos: int) -> bool:
    """Check if a counter expression could start at this position."""
    if pos >= len(text):
        return False
    char = text[pos]
    return char in KANJI_NUMBERS or char in DIGIT_VALUES or char.isdigit()
