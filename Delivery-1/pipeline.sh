#!/bin/bash

# clean and tokenize the dataset
python main.py --clean_data True --raw_data_dir 2021-01 --f_name dataset

# export ngrams (i.e., bigrams, trigrams, etc..)
python main.py --export_ngrams True --f_name dataset

# extract collocations based on frequency
python main.py --method chi_square --f_name dataset
