import json
import math
from tqdm import tqdm


class DiffMeanVariance:
    def __init__(self, data, window_size = 3):
        self.data = data                # Dataset in json format
        self.collocations_offset = {}   # to store collocations with offset
        self.collocations_mean = {}     # to store collocations mean
        self.collocations_variance = {} # to store collocations variance
        self.collocations_std_dev = {}  # to store collocations standard deviation
        # Execute the functions to find diff mean and variance
        self.find_offsets(window_size) 
        self.find_means()
        self.find_variance()
        self.find_std_dev()


    def find_offsets(self, window_size):
        # Find the offset of the collocations with given window size
        for sent in tqdm(self.data.values()):
            splitted = sent.split(' ')
            sentence_size = len(sent.split(' '))
            for i in range(sentence_size):
                for j in range(max(0,i-3), min(i+window_size, sentence_size-1)+1):
                    if i==j:continue
                    self.collocations_offset.setdefault(splitted[i] + ' ' + splitted[j], []).append(j-i)


    def find_means(self):
        # Calculate mean based on the offset over the occurence of the word
        for key, value in self.collocations_offset.items():
            self.collocations_mean[key] = sum(value)/len(value)


    def find_variance(self):
        # Calculate variance based on the offset over the occurence of the word
        for key, value in self.collocations_offset.items():
            if len(value) == 1:
                continue
            mean = self.collocations_mean[key]
            self.collocations_variance[key] = sum([(offset-mean)**2 for offset in value])/(len(value)-1)


    def find_std_dev(self):
        # Calculate the standard deviation from the variance
        for key, value in self.collocations_variance.items():
            self.collocations_std_dev[key] = math.sqrt(value)


    def export_collocation_by_diff_mean_var(self):
        # Extract the collocations with the standard deviation less than 0.5
        # {'word1 word2': {'mean': mean, 'variance': variance, 'std_dev': std_dev}}
        collocations_result ={}
        for key, value in self.collocations_std_dev.items():
            if value < 0.5:
                collocations_result[key] = {'mean': self.collocations_mean[key], 'variance': self.collocations_variance[key], 'std_dev': value}  

        # Export the results
        with open('data/diff_mean_var_collocation.json', 'w', encoding='utf-8') as f:
            json.dump(collocations_result, f, ensure_ascii=False, indent=4)
