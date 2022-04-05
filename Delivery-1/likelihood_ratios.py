import math
from collections import Counter
from tqdm import tqdm
from scipy.stats import binom
import json
import sys


class LikelihoodRatios:
    def __init__(self, data, bigrams):
        self.data = data
        self.bigrams = bigrams

    def calculate_p(self, c2, N): # Calculates P(w2|w1) == P(w2|!(w1))
        return c2 / N

    def calculate_p1(self, c1_c2, c1): # Calculates P(w2|w1)
        return c1_c2 / c1

    def calculate_p2(self, c1, c2, c1_c2, N): # Calculates P(w2|!(w1))
        return (c2 - c1_c2) / (N - c1)

    def calculate_likelihood_h1(self, c1, c2, c1_c2, N): # Calculates likelihood of Hypothesis 1
        p = self.calculate_p(c2, N)
        return binom.pmf(c1_c2, c1, p) * binom.pmf(c2-c1_c2, N - c1, p)

    def calculate_likelihood_h2(self, c1, c2, c1_c2, N): # Calculates likelihood of Hypothesis 2
        p1 = self.calculate_p1(c1_c2, c1)
        p2 = self.calculate_p2(c1, c2, c1_c2, N)
        return binom.pmf(c1_c2, c1, p1) * binom.pmf(c2-c1_c2, N - c1, p2)

    def calculate_likelihood(self, k, n, x):
        val1 = pow(math.log(x), math.log(k))
        val2 = pow(math.log(1-x), math.log(n-k))
        print(val1)
        return 1

    def get_ratio(self, c1, c2, c1_c2, N):  # Calculates likelihood Ratio * (-2)
        lh1 = self.calculate_likelihood_h1(c1, c2, c1_c2, N)
        lh2 = self.calculate_likelihood_h2(c1, c2, c1_c2, N)
        ratio = lh1 / lh2
        if ratio == int(0):
            ratio_log = 0
        else:
            ratio_log = math.log(ratio)
        return round(-2 * ratio_log, 2)

    def count_words(self): # Function to count the number of words in the dataset
        words = Counter()
        for val in tqdm(self.data.values()):
            splitted = val.split()
            words.update(splitted)
        dataset_size = sum(words.values())
        return (words, dataset_size)

    def get_bigrams_ratios(self):
        words, N = self.count_words()
        bigram_len = len(self.bigrams)
        words_ratios = []  # Contains list of
        #  {"ratio": float, "c1": int, "c2": int, "c1_c2": int, "w1": string, "w2": string}
        counter = 0
        for bi in self.bigrams:
            w1_w2 = bi.split(" ")
            c1_c2 = self.bigrams[bi]
            w1, w2 = (w1_w2[0], w1_w2[1])
            c1, c2 = words[w1], words[w2]
            ratio = self.get_ratio(c1, c2, c1_c2, N)
            words_ratios.append({"ratio": ratio, "c1": c1, "c2": c2, "c1_c2": c1_c2, "w1": w1, "w2": w2})
            counter += 1
            sys.stdout.write("Progress: %d%%   \r" % ((counter/bigram_len)*100) )
            sys.stdout.flush()
        return words_ratios
            

    def export_collocation_by_likelihood_ratios(self, n=20):
        words_ratio = self.get_bigrams_ratios()
        sorted_words_ratio = sorted(words_ratio, key=lambda d: d['ratio'], reverse=True)
        collocations = {}
        for w in range(n):
            w1, w2 = (sorted_words_ratio[w]['w1'], sorted_words_ratio[w]['w2'])
            c1, c2, c1_c2 = (sorted_words_ratio[w]['c1'],sorted_words_ratio[w]['c2'], sorted_words_ratio[w]['c1_c2'])
            ratio = sorted_words_ratio[w]['ratio']
            collocations[w1 + " " + w2] = [ratio, c1, c2, c1_c2, w1, w2]

        with open(f'data/likelihood_ratio_collocation_top_{n}.json', 'w', encoding='utf-8') as f:
            json.dump(collocations, f, ensure_ascii=False, sort_keys=True, indent=4)
