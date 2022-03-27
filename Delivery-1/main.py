import json
import os
import argparse
from zemberek import TurkishTokenizer


def parse_data(files, args):
    data = {}
    for f_name in files:
        with open(args.raw_data_dir + '/' + f_name, encoding='utf-8') as f:
            f_number = f_name.split('.')[0]
            data[f_number] = json.load(f)['ictihat']

    return data


def clean_data(data):
    tokenizer = TurkishTokenizer.DEFAULT

    for key in data:
        tokens = tokenizer.tokenize(data[key])

        filtered_tokens = []
        for token in tokens:
            if token.type_.name != 'Word': continue
            filtered_tokens.append(token.content.lower())

        data[key] = ' '.join(filtered_tokens)

    return data


def main(args):
    if args.clean_data:
        data = parse_data(os.listdir(args.raw_data_dir), args)
        data = clean_data(data)
        with open('clean_data.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=4)
    else:
        pass


def parse_args():
    parser = argparse.ArgumentParser("Collocation Extractor")
    parser.add_argument('--clean_data', type=bool, default=False, help='clean the raw data')
    parser.add_argument('--raw_data_dir', type=str, default='2021-01', help='raw dataset directory')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
