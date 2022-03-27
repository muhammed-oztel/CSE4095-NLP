from zemberek import TurkishMorphology
from tqdm import tqdm
import json

morphology = TurkishMorphology.create_with_defaults()

def export_collocation_by_frequency(bigrams, n=20):
    collocations = {}

    for bigram in tqdm(bigrams.keys()):
        tokens = bigram.split()
        pos_tags = (get_pos_tag(tokens[0]), get_pos_tag(tokens[1]))
        if pos_tags[0] == 'N' and pos_tags[1] == 'N':
            collocations[bigram] = bigrams[bigram]
        elif pos_tags[0] == 'A' and pos_tags[1] == 'N':
            collocations[bigram] = bigrams[bigram]


    with open('frequency_collocation.json', 'w', encoding='utf-8') as f:
        json.dump(collocations, f, ensure_ascii=False, sort_keys=True, indent=4)

    sort_orders = sorted(collocations.items(), key=lambda x: x[1], reverse=True)[:n]
    for item in sort_orders:
        print(str(item[0]) + ' ' + str(item[1]))


def get_pos_tag(token):
    res = morphology.analyze(token.strip())
    tags_str = str(res)

    if ':Noun' in tags_str:
        return 'N'
    
    if ':Adj' in tags_str:
        return 'A'
