import json


# Frequency
from frequency import Frequency
with open('data/frequency_collocation_bigram_top_20.json', encoding='utf-8') as fr:
    results = json.load(fr)

frequency = Frequency({}, {})
print("{:<10} {:<15} {:<15} {:<10}".format('C(w1,w2)', 'w1', 'w2', 'TP'))
for key in results:
    ngram = key[0].split()
    freq = key[1]
    tag_pattern = ''.join([frequency.get_pos_tag(token) for token in ngram])
    print("{:<10} {:<15} {:<15} {:<10}".format(freq, ngram[0], ngram[1], tag_pattern))
