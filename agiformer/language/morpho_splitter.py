# agiformer/language/morpho_splitter.py

# Developer: inkbytefo
# Modified: 2025-11-06

import re
import logging

logger = logging.getLogger(__name__)

class RegexSplitter:
    """
    A lightweight, regex-based morphological analyzer for Turkish.
    This replaces heavy external dependencies with a fast, rule-based approach
    that focuses on common Turkish suffixes while maintaining compatibility
    with the preprocessing pipeline.
    """

    # Common Turkish suffixes and their patterns
    SUFFIX_PATTERNS = {
        'Acc': r'(?:yi|yu|ye|yü|yi|yu|ye|yü)$',  # accusative
        'Dat': r'(?:ya|ye|ya|ye)$',              # dative
        'Loc': r'(?:da|de|ta|te)$',               # locative
        'Abl': r'(?:dan|den|tan|ten)$',           # ablative
        'Gen': r'(?:ın|in|un|ün)$',               # genitive
        'P1sg': r'(?:ım|im|um|üm)$',             # 1st person singular
        'P2sg': r'(?:ın|in|un|ün)$',             # 2nd person singular
        'P3sg': r'(?:ı|i|u|ü)$',                 # 3rd person singular
        'P1pl': r'(?:ımız|imiz|umuz|ümüz)$',     # 1st person plural
        'P2pl': r'(?:ınız|iniz|unuz|ünüz)$',     # 2nd person plural
        'P3pl': r'(?:ları|leri)$',               # 3rd person plural
        'Pl': r'(?:lar|ler)$',                   # plural
        'Verb': r'(?:yor|ıyor|iyor|üyor|ıyor|iyor|üyor)$',  # progressive
        'Past': r'(?:tı|ti|tu|tü|dı|di|du|dü|ti|ti|tu|tü)$',  # past tense
        'Fut': r'(?:acak|ecek)$',                # future
        'Cond': r'(?:sa|se)$',                   # conditional
        'Imp': r'(?:sın|sin|sun|sün)$',          # imperative
        'Inf': r'(?:mak|mek)$',                  # infinitive
        'Neg': r'(?:ma|me)$',                    # negative
    }

    # Vowel harmony patterns for better root extraction
    VOWEL_HARMONY = {
        'a': ['a', 'ı', 'u'],
        'e': ['e', 'i', 'ü'],
        'ı': ['a', 'ı', 'u'],
        'i': ['e', 'i', 'ü'],
        'o': ['a', 'ı', 'u'],
        'ö': ['e', 'i', 'ü'],
        'u': ['a', 'ı', 'u'],
        'ü': ['e', 'i', 'ü']
    }

    def __init__(self):
        logger.info("Initializing RegexSplitter...")
        self._initialized = True

    def split_sentence(self, sentence: str) -> dict:
        """
        Analyzes a sentence and splits each word into its root and suffixes
        using regex-based morphological analysis.

        Args:
            sentence: The input sentence string.

        Returns:
            A dictionary containing the analysis results.
        """
        words = sentence.split()
        output = {"kelimeler": []}

        for word in words:
            root, suffixes = self._analyze_word(word)
            output["kelimeler"].append({
                "kök": root,
                "ekler": suffixes
            })

        return output

    def _analyze_word(self, word: str) -> tuple[str, list[str]]:
        """
        Analyzes a single word to extract root and suffixes.

        Args:
            word: The word to analyze.

        Returns:
            A tuple of (root, [suffixes]).
        """
        word = word.strip()
        if not word:
            return "", []

        # Try to find suffixes from longest to shortest
        remaining = word
        found_suffixes = []

        # Sort patterns by length (longest first) to handle overlapping suffixes
        sorted_patterns = sorted(self.SUFFIX_PATTERNS.items(),
                               key=lambda x: len(x[1]), reverse=True)

        for suffix_name, pattern in sorted_patterns:
            match = re.search(pattern, remaining, re.IGNORECASE)
            if match:
                suffix = match.group()
                # Check if this suffix respects vowel harmony with the preceding syllable
                if self._check_vowel_harmony(remaining[:-len(suffix)], suffix):
                    found_suffixes.append(suffix)
                    remaining = remaining[:-len(suffix)]
                    # Limit to 3 suffixes to avoid over-analysis
                    if len(found_suffixes) >= 3:
                        break

        # If no suffixes found, the whole word is the root
        if not found_suffixes:
            return word, []

        return remaining, found_suffixes[::-1]  # Reverse to get root-first order

    def _check_vowel_harmony(self, stem: str, suffix: str) -> bool:
        """
        Performs a basic vowel harmony check.

        Args:
            stem: The stem part of the word.
            suffix: The suffix to check.

        Returns:
            True if vowel harmony is respected, False otherwise.
        """
        if not stem or not suffix:
            return True

        # Get the last vowel of the stem
        stem_vowels = [c for c in stem if c in self.VOWEL_HARMONY]
        suffix_vowels = [c for c in suffix if c in self.VOWEL_HARMONY]

        if not stem_vowels or not suffix_vowels:
            return True

        last_stem_vowel = stem_vowels[-1]
        first_suffix_vowel = suffix_vowels[0]

        # Check if the suffix vowel is compatible with the stem's harmony group
        return first_suffix_vowel in self.VOWEL_HARMONY[last_stem_vowel]
