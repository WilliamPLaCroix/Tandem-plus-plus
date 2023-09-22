from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import json

def compile_CEFR_setlist()-> list:
    """
    Grouped data structure format:
    groups[group number][column][row]
    group number: 0-5, indexed to [A1, A2, B1, B2, C1, C2]
    column: 'CEFR', 'Text'
    row: zero-indexed
    example: groups[0]['Text'][0]
    pulls first text from A1 group
    """
    with open('./cefr-dataset/data.csv', 'r', encoding="utf-8") as file:
        gb = pd.read_csv(file, index_col=0).applymap(word_tokenize).groupby('CEFR')
        groups = [gb.get_group(group) for group in gb.groups]
    return groups



def build_constraint_lists():
    with open('constraint_lists.json', 'r', encoding="utf-8") as f:
        constraint_lists = json.load(f)
    full_A1_word_list = {word.lower() for unit in constraint_lists['A1 Full List'] for word in constraint_lists['A1 Full List'][unit]}
    high_frequency_A1_word_list = {word.lower() for unit in constraint_lists['A1 High Frequency List'] for word in constraint_lists['A1 High Frequency List'][unit]}
    return full_A1_word_list, high_frequency_A1_word_list

def build_dataset():

    full_A1_word_list, high_frequency_A1_word_list = build_constraint_lists()
    data = pd.concat(compile_CEFR_setlist())
    data['% High Frequency'] = np.nan
    data['% Full List'] = np.nan
    for i, text in enumerate(data['Text']):
        text = data['Text'][i].copy()
        for word in text:
            if word.isalpha() == False:
                text.remove(word)
        test_text = text
        for word in test_text:
            if word.isalpha() == False:
                text.remove(word)
        text_length = len(test_text)
        for word in text:
            if word.lower() not in full_A1_word_list:
                test_text.remove(word)
        data['% Full List'][i] = len(test_text)/text_length*100
        for word in test_text:
            if word.lower() not in high_frequency_A1_word_list:
                test_text.remove(word)
        data['% High Frequency'][i] = len(test_text)/text_length*100
    return data


def get_group_means():
    data = build_dataset()
    groups = data.groupby('CEFR')
    high_frequency_means = groups['% High Frequency'].mean()
    full_list_means = groups['% Full List'].mean()
    high_frequency_means_list = high_frequency_means.to_dict()
    full_list_means_list = full_list_means.to_dict()
    print(high_frequency_means_list)
    print(full_list_means_list)
    return high_frequency_means_list, full_list_means_list

import matplotlib.pyplot as plt

def plot_percentage_curves():
    high_frequency_means_list, full_list_means_list = get_group_means()
    list1 = high_frequency_means_list.items()
    list2 = full_list_means_list.items()
    x1, y1 = zip(*list1)
    x2, y2 = zip(*list2)

    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.legend(['High Frequency', 'Full List'])
    plt.xlabel('CEFR Level')
    plt.ylabel('Percentage of Words in Text that are in List')
    plt.show()

    fit1 = np.polyfit(range(1, 7), y1, 1)
    fit2 = np.polyfit(range(1, 7), y2, 1)
    plt.plot(range(1, 7), fit1[0] * range(1, 7) + fit1[1], color='red')
    plt.plot(range(1, 7), fit2[0] * range(1, 7) + fit2[1], color='red')
    plt.show()

def word_list_percentages(text):

    full_A1_word_list, high_frequency_A1_word_list = build_constraint_lists()
    for word in text:
        if word.isalpha() == False:
            text.remove(word)
    test_text = text
    for word in test_text:
        if word.isalpha() == False:
            text.remove(word)
    text_length = len(test_text)
    for word in text:
        if word.lower() not in full_A1_word_list:
            test_text.remove(word)
    full_list_percentage = len(test_text)/text_length*100
    for word in test_text:
        if word.lower() not in high_frequency_A1_word_list:
            test_text.remove(word)
    high_frequency_percentage = len(test_text)/text_length*100

    return full_list_percentage, high_frequency_percentage

def CEFR_level_guesser(test_text):

    high_frequency_percentage, full_list_percentage = word_list_percentages(test_text)
    hyperparameters = [[-3.2714356668112687, 63.878587858073395],
                        [-2.6896079850794714, 81.13135899087996]]
    CEFRs = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
    m1, b1 = hyperparameters[0]
    m2, b2 = hyperparameters[1]
    x1 = round((high_frequency_percentage-b1)/m1)
    x2 = round((full_list_percentage-b2)/m2)

    if x1 < 0:
        level_guess1 = CEFRs[0]
    elif x1 > 5:
        level_guess1 = CEFRs[5]
    else:
        level_guess1 = CEFRs[x1]

    if x2 < 0:
        level_guess2 = CEFRs[0]
    elif x2 > 5:
        level_guess2 = CEFRs[5]
    else:
        level_guess2 = CEFRs[x2]

    if level_guess1 == level_guess2:
        return level_guess1
    elif x1 > x2:
        return f'{level_guess2}-{level_guess1}'
    else:
        return f'{level_guess1}-{level_guess2}'
