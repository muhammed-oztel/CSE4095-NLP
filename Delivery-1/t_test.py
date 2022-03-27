from collections import Counter
from decimal import DivisionByZero
from tqdm import tqdm
import math
import json


def export_collocation_by_t_test(data, bigrams, n=20):
    collocations = {}
    word_probs = get_word_freqs(data)
    dataset_size = sum(word_probs.values())

    for bigram in tqdm(bigrams.keys()):
        tokens = bigram.split()
        pop_mean = (word_probs[tokens[0]] / dataset_size) * (word_probs[tokens[1]] / dataset_size)
        sample_mean = bigrams[bigram] / dataset_size
        try:
            t_stat = (sample_mean - pop_mean) / math.sqrt(sample_mean / dataset_size)
        except DivisionByZero:
            t_stat = 0
        
        collocations[bigram] = round(t_stat, 2)

    with open('ttest_collocation.json', 'w', encoding='utf-8') as f:
        json.dump(collocations, f, ensure_ascii=False, sort_keys=True, indent=4)

    sort_orders = sorted(collocations.items(), key=lambda x: x[1], reverse=True)[:n]
    for item in sort_orders:
        print(str(item[0]) + ' ' + str(item[1]))


def get_word_freqs(data):
    words = Counter()
    for val in tqdm(data.values()):
        splitted = val.split()
        words.update(splitted)

    return words
