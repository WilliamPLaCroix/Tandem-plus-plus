import re
import string

import pandas as pd
import textgrid as tg  # https://github.com/nltk/nltk_contrib/blob/master/nltk_contrib/textgrid.py
from nltk.tokenize import word_tokenize


def tokenize_cefr_texts() -> list:
    """
    Grouped data structure format:
    groups[group number][column][row]
    group number: 0-5, indexed to [A1, A2, B1, B2, C1, C2]
    column: 'CEFR', 'Text'
    row: zero-indexed
    example: groups[0]['Text'][0]
    pulls first text from A1 group
    """
    with open("./cefr-dataset/data.csv", "r", encoding="utf-8") as file:
        gb = pd.read_csv(file, index_col=0).applymap(word_tokenize).groupby("CEFR")
        groups = [gb.get_group(group) for group in gb.groups]
    return groups


def tokenize_textgrid(filename) -> list:
    """
    Reads in TextGrid file, extracting transcript from grid.tiers[0]
    cleans off [] from (sic?) words,
    returns tokenized list >>> no punctuation, <P> indicates pause
    """
    pattern = re.compile("[\[\]]+")
    fp = open(filename, "r", encoding="utf-8").read()
    grid = tg.TextGrid(fp)
    tier_0 = grid.tiers[0]
    full_text = " ".join(
        [
            text
            for (start, stop, text) in tier_0.simple_transcript
            if text != "<HÄSITATION>"
        ]
    )
    cleaned = pattern.sub("", full_text).split()
    return cleaned
