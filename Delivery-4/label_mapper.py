import json
import os
from tqdm import tqdm

files = os.listdir('./data/2021-01')

labels = {}
with open('data/label_map.json', 'r', encoding='utf-8') as f:
    labels = json.load(f)

instance_labels = {}

for f_name in tqdm(files):
    instance = {}
    
    k = f_name.split('.')[0]
    try:
        a = int(k)
    except:
        continue

    with open('./data/2021-01/' + f_name, 'r', encoding='utf-8') as f:
        instance = json.load(f)
    
    mahkeme_label = instance['Mahkemesi']

    if mahkeme_label == '':
        instance_labels[f_name.split('.')[0]] = 'EMPTY'
        continue

    found = False
    for label in labels:
        for l in labels[label]:
            if l in mahkeme_label.strip():
                instance_labels[f_name.split('.')[0]] = label
                found = True
    
    if not found:
        instance_labels[f_name.split('.')[0]] = 'OTHER'

with open('data/labels.json', 'w', encoding='utf-8') as f:
    json.dump(instance_labels, f, indent=4, ensure_ascii=False)
