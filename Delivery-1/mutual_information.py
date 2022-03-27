from base64 import decode
from collections import Counter
import json
import numpy as np
import math
from tqdm import tqdm

class MutualInformation:
    def __init__(self, data, bigrams):
        self.data = data
        self.bigrams = bigrams

    # Function to load the data and bigrams
    def load_data(self):
        with open('clean_data.json', 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        with open('bigrams.json', 'r', encoding='utf-8') as f:
            self.bigrams = json.load(f)

    # Function to count the number of words in the dataset
    def count_words(self, data):
        self.words = Counter()
        for val in tqdm(data.values()):
            splitted = val.split()
            self.words.update(splitted)
        self.dataset_size = sum(self.words.values())

    # Function to calculate the mutual information
    def mutual_information(self, word1, word2):
        # Calculate the probability of the two words
        prob_word1 = self.words[word1] / self.dataset_size
        prob_word2 = self.words[word2] / self.dataset_size
        # Calculate the probability of the bigram
        prob_bigram = self.bigrams[word1+ ' ' +word2] / self.dataset_size
        # Calculate the mutual information
        return math.log(prob_bigram / (prob_word1 * prob_word2), 2)

    def export_collocation_by_pmi(self):
        self.count_words(self.data)

        # Calculate the mutual information for all bigrams
        mutual_info = {}
        for bigram in tqdm(self.bigrams.keys()):
            mutual_info[bigram] = self.mutual_information(bigram.split()[0], bigram.split()[1])

        # Extract all bigrams to a json file   
        with open('mutual_information.json', 'w', encoding='utf-8') as f:
            json.dump(mutual_info, f, ensure_ascii=False, sort_keys=True, indent=4)

        # # Extract top 20 bigrams to a json file
        # top_20 = sorted(mutual_info.items(), key=lambda x: x[1], reverse=True)[:20]
        # with open('top_20.json', 'w', encoding='utf-8') as f:
        #     json.dump(top_20, f, ensure_ascii=False, indent=4)

