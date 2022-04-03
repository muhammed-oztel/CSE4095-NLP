from collections import Counter
from tqdm import tqdm
import json


class PearsonChiSquareTest:
    def __init__(self, data, bigrams):
        self.data = data
        self.bigrams = bigrams
        self.count_words()

    # Function to count the number of words in the dataset
    def count_words(self):
        self.words = Counter()
        for val in tqdm(self.data.values()):
            splitted = val.split()
            self.words.update(splitted)
        self.dataset_size = sum(self.words.values())

    def get_word_count(self, w1, w2, c_w1_w2):
        c_w1 = self.words[w1]
        c_w2 = self.words[w2]
        cell_11 = c_w1_w2
        cell_12 = c_w2 - c_w1_w2
        cell_21 = c_w1 - c_w1_w2
        cell_22 = self.dataset_size - cell_12 - cell_21

        return cell_11, cell_12, cell_21, cell_22

    def export_collocations_by_chi_square(self, n=20):
        collocations = {}

        for bigram in self.bigrams:
            w1, w2 = bigram.split()
            cell_11, cell_12, cell_21, cell_22 = self.get_word_count(
                w1, w2, self.words[bigram])
            chi_square = (self.dataset_size * ((cell_11 * cell_22 - cell_12 * cell_21) ** 2)) / (
                (cell_11 + cell_12) * (cell_11 + cell_21) * (cell_12 + cell_22) * (cell_21 + cell_22))
            collocations[bigram] = chi_square

        with open('chi_square_collocation.json', 'w', encoding='utf-8') as f:
            json.dump(collocations, f, ensure_ascii=False,
                      sort_keys=True, indent=4)

        sort_orders = sorted(collocations.items(),
                             key=lambda x: x[1], reverse=True)[:n]
        for item in sort_orders:
            print(str(item[0]) + ' ' + str(item[1]))
