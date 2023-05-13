import pandas as pd
from nltk.tokenize import word_tokenize


def tokenize_dataset() -> list:
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
        groups = [
            gb.get_group(group)  # ignore the linter error, it's just jealous.
            for group in pd.read_csv(file, index_col=0)  # df with columns CEFR, Text
            .applymap(word_tokenize)  # tokenize each text
            .groupby("CEFR")  # separate into [A1, A2, B1, B2, C1, C2]
            .groups  # pull group items
        ]
    return groups
