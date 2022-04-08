from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os
import re

# loading dataset
data_path = "2021-01"

list_of_files = os.listdir(data_path)

# dropping duplicate filenames
sorted_list_of_files = sorted([int(i.split('.')[0]) for i in list_of_files if '(' not in i])
sorted_list_of_files = [f"{i}.json" for i in sorted_list_of_files]

# to read the json files
def read_files(file_name):
    with open(f"{data_path}/{file_name}", 'rb') as f:
        an_instance = json.load(f)
    return an_instance['ictihat']

# cleaning text from paranthesis
cleaned_texts = []

with ThreadPoolExecutor() as executor:
    texts = list(executor.map(read_files, sorted_list_of_files))

for text in tqdm(texts):
    cleaned_text = " ".join(re.sub(r'\([^)]*\)', '', text).strip().split())
    cleaned_texts.append(cleaned_text)

# saving cleaned texts
pd.DataFrame({'id':np.arange(1, len(cleaned_texts)+1),
              'text':np.array(cleaned_texts)}).to_csv(f"cleaned_data_for_ctm.csv", index=False)