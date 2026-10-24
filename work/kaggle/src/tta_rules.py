"""Deterministic surface forms only; never alter numerical values or units."""
import re
from collections import Counter
from decimal import Decimal

NUMBER = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+'
NUMBERS = re.compile(NUMBER)
AMOUNT = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
PREFIX = re.compile(r'\$(' + AMOUNT + r')(?![\d.,\w])(?!(?:\s+|-)(?:dollars|hundred|thousand|million|billion|trillion)\b)', re.I)
WORDS = re.compile(r'(?<![\w.$])(' + AMOUNT + r') dollars\b(?![\w\x27’])')
GROUPED = re.compile(r'(?<![\w.,])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w.,])')
SPACE = re.compile(r' +([?!;:])')
PRICE_ZERO = re.compile(r'\$(' + AMOUNT + r')(?![\d.,\w])')


def numbers(text):
    return Counter(Decimal(m.group().replace(',', '')) for m in NUMBERS.finditer(text))


def dollar_words(text):
    # Require a token boundary after the full amount: no $5k or $5 million.
    return PREFIX.sub(lambda m: m[1] + ' dollars', text)


def dollar_symbol(text):
    return WORDS.sub(lambda m: '$' + m[1], text)


def ungroup(text):
    return GROUPED.sub(lambda m: m[0].replace(',', ''), text)


def punctuation(text):
    return SPACE.sub(r'\1', text)


def price_trim(text):
    def trim(m):
        value=m[1]
        return '$'+(value.rstrip('0').rstrip('.') if '.' in value else value)
    return PRICE_ZERO.sub(trim, text)


RULES = {'dollar-words': dollar_words, 'dollar-symbol': dollar_symbol,
         'ungroup': ungroup, 'punctuation': punctuation, 'price-trim': price_trim}


def apply(text, name):
    result = RULES[name](text)
    if numbers(result) != numbers(text):
        raise ValueError('TTA changed the multiset of numbers')
    return result
