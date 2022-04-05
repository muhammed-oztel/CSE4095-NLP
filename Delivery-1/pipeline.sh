#!/bin/bash

# clean and tokenize the dataset
python main.py --clean_data True --raw_data_dir 2021-01 --f_name dataset

# export ngrams (i.e., bigrams, trigrams, etc..)
python main.py --export_ngrams True --f_name dataset

# extract collocations based on frequency
python main.py --method frequency --f_name dataset

# extract collocations based on mutual information
python main.py --method pmi --f_name dataset

# extract collocations based on t-test
python main.py --method t_test --f_name dataset

# extract collocations based on diff mean variance
python main.py --method diff_mean_var --f_name dataset

# extract collocations based on chi-square test
python main.py --method chi_square --f_name dataset

# extract collocations based on likelihood ratios
python main.py --method likelihood_ratios --f_name dataset
