from collections import Counter
from decimal import DivisionByZero
from tqdm import tqdm
import math
import json


class TTest:
    def __init__(self, data, bigrams):
        self.data = data
        self.bigrams = bigrams


    def export_collocation_by_t_test(self, n=20): # method to export the collocations by t-test
        collocations = {}
        word_probs = self.get_word_freqs()
        dataset_size = sum(word_probs.values())

        for bigram in tqdm(self.bigrams.keys()):
            tokens = bigram.split()
            pop_mean = (word_probs[tokens[0]] / dataset_size) * (word_probs[tokens[1]] / dataset_size)
            sample_mean = self.bigrams[bigram] / dataset_size
            try:
                t_stat = (sample_mean - pop_mean) / math.sqrt(sample_mean / dataset_size)
            except DivisionByZero:
                t_stat = 0
            
            collocations[bigram] = round(t_stat, 2)

        with open('data/t_test_collocation.json', 'w', encoding='utf-8') as f:
            json.dump(collocations, f, ensure_ascii=False, sort_keys=True, indent=4)

        sort_orders = sorted(collocations.items(), key=lambda x: x[1], reverse=True)[:n]
        with open(f'data/t_test_collocation_top_{n}.json', 'w', encoding='utf-8') as f:
            json.dump(sort_orders, f, ensure_ascii=False, sort_keys=True, indent=4)


    def get_word_freqs(self): # returns a dictionary of word frequencies
        words = Counter()
        for val in tqdm(self.data.values()):
            splitted = val.split()
            words.update(splitted)

        return words
