import json
import math
   
class MeanVariance:
    def __init__(self, data):
        self.data = data
        self.collocations_offset = {}
        self.collocations_mean = {}
        self.collocations_variance = {}
        self.collocations_std_dev = {}
        self.find_offsets()
        self.find_means()
        self.find_variance()
        self.find_std_dev()

    def load_data(self, path='clean_data.json'):
        with open(path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def find_offsets(self, window_size = 5):
        for sent in self.data.values():
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

    def export_collocations(self, extract_path = 'mean_variance.json'):
        # Generate a dictionary in given format
        # {'word1 word2': {'mean': mean, 'variance': variance, 'std_dev': std_dev}}
        collocation_results = {}
        for key, value in self.collocations_std_dev.items():
            collocation_results[key] = {'mean': self.collocations_mean[key], 'variance': self.collocations_variance[key], 'std_dev': value}  

        # Extract collocations to a json file
        with open(extract_path, 'w', encoding='utf-8') as f:
            json.dump(collocation_results, f, ensure_ascii=False, sort_keys=True, indent=4)


### Simple Example Run ###
# mv = MeanVariance(data = {
#        1: "she knocked on his door", 
#         2:"they knocked at the door", 
#         3:"100 women knocked on Donaldson's asd asd door", 
#         4:"a man knocked on the metal front door",
# })
# mv.export_collocations('mean_var_small.json')