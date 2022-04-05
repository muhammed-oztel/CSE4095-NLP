import json
import os
import argparse
import random
from typing import Counter
from zemberek import TurkishTokenizer
from frequency import Frequency
from t_test import export_collocation_by_t_test
from hypothesis_testing_of_differences import calculate_t_values
from tqdm import tqdm
from mutual_information import MutualInformation
from mean_variance import MeanVariance
from likelihood_ratios import get_top_bigrams
from chi_square_test import PearsonChiSquareTest


def parse_data(files, args):
    data = {}
    pbar = tqdm(files)
    for f_name in pbar:
        pbar.set_description(f'parsing: {f_name}')
        with open('data/' + args.raw_data_dir + '/' + f_name, encoding='utf-8') as f:
            f_number = f_name.split('.')[0]
            data[f_number] = json.load(f)['ictihat']

    return data


def export_data(data, args):
    tokenizer = TurkishTokenizer.DEFAULT

    pbar = tqdm(data)
    for key in pbar:
        pbar.set_description(f'cleaning file {key}')
        tokens = tokenizer.tokenize(data[key])

        filtered_tokens = []
        for token in tokens:
            if token.type_.name != 'Word': continue
            filtered_tokens.append(token.content.lower())

        data[key] = ' '.join(filtered_tokens)

    with open(f'data/{args.f_name}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=4)


def export_ngrams(n, args):
    data = {}
    with open(f'data/{args.f_name}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    ngrams = Counter()
    pbar = tqdm(data)
    for key in pbar:
        pbar.set_description(f'counting {n}-grams for file {key}')
        splitted = data[key].split()
        for i in range(len(splitted)-n+1):
            ngrams.update([' '.join(splitted[i:i+n])])

    with open(f'data/{n}grams_{args.f_name}.json', 'w', encoding='utf-8') as f:
        json.dump(ngrams, f, ensure_ascii=False, sort_keys=True, indent=4)


def main(args):
    if args.clean_data:
        data = parse_data(os.listdir('data/' + args.raw_data_dir), args)
        export_data(data, args)

    elif args.export_ngrams:
        export_ngrams(2, args)
        export_ngrams(3, args)

    else:
        with open(f'data/{args.f_name}.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        with open(f'data/2grams_{args.f_name}.json', 'r', encoding='utf-8') as f:
            bigrams = json.load(f)

        with open(f'data/3grams_{args.f_name}.json', 'r', encoding='utf-8') as f:
            trigrams = json.load(f)

        if args.method == 'frequency':
            frequency = Frequency(bigrams, trigrams)
            frequency.export_collocations_by_frequency(type='bigram', n=20)
            frequency.export_collocations_by_frequency(type='trigram', n=20)

        elif args.method == 'pmi':
            mutual_information = MutualInformation(data, bigrams)
            mutual_information.export_collocation_by_pmi()

        elif args.method == 't_test':
            export_collocation_by_t_test(data, bigrams, n=20)

        elif args.method == 'meanvar':
            mv = MeanVariance(data)
            mv.export_collocations()

        elif args.method == "hypo":
            calculate_t_values(bigrams)

        elif args.method == 'chi_square':
            print('\n===== CHI SQUARE TEST =====')
            chi_square = PearsonChiSquareTest(data, bigrams)
            chi_square.export_collocations_by_chi_square()

        elif args.method == "likeratio":
            get_top_bigrams(bigrams, data)


def parse_args():
    parser = argparse.ArgumentParser("Collocation Extractor")
    parser.add_argument('--clean_data', type=bool, default=False, help='clean the raw data')
    parser.add_argument('--raw_data_dir', type=str, default='2021-01', help='raw dataset directory')
    parser.add_argument('--f_name', type=str, default='dataset', help='file name')
    parser.add_argument('--export_ngrams', type=bool, default=False, help='export ngrams (i.e., n=2,3,..)')
    parser.add_argument('--method', type=str, default='frequency', help='method to extract collocations')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
