from zemberek import TurkishMorphology
from tqdm import tqdm
import json


class Frequency:
    def __init__(self, bigrams, trigrams):
        self.bigrams = bigrams
        self.trigrams = trigrams
        self.morphology = TurkishMorphology.create_with_defaults()


    def export_collocation_by_frequency(self, type='bigram', n=20):
        collocations = {}

        if type == 'bigram':
            for bigram in tqdm(self.bigrams.keys()):
                tokens = bigram.split()
                pos_tags = (self.get_pos_tag(tokens[0]), self.get_pos_tag(tokens[1]))
                if None in pos_tags: continue
                if pos_tags[0] + pos_tags[1] in ['NN', 'AN']:
                    collocations[bigram] = self.bigrams[bigram]

        elif type == 'trigram':
            for trigram in tqdm(self.trigrams.keys()):
                tokens = trigram.split()
                pos_tags = (self.get_pos_tag(tokens[0]), self.get_pos_tag(tokens[1]), self.get_pos_tag(tokens[2]))
                if None in pos_tags: continue
                if pos_tags[0] + pos_tags[1] + pos_tags[2] in ['AAN', 'ANN', 'NAN', 'NNN', 'NPN']:
                    collocations[trigram] = self.trigrams[trigram]

        with open(f'data/frequency_collocation_{type}.json', 'w', encoding='utf-8') as f:
            json.dump(collocations, f, ensure_ascii=False, sort_keys=True, indent=4)

        sort_orders = sorted(collocations.items(), key=lambda x: x[1], reverse=True)[:n]
        with open(f'data/frequency_collocation_{type}_top_{n}.json', 'w', encoding='utf-8') as f:
            json.dump(sort_orders, f, ensure_ascii=False, sort_keys=True, indent=4)


    def get_pos_tag(self, token):
        res = self.morphology.analyze(token.strip())
        tags_str = str(res)

        if ':Noun' in tags_str:
            return 'N'
        
        if ':Adj' in tags_str:
            return 'A'
