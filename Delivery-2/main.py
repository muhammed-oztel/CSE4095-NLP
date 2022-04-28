import warnings
warnings.simplefilter('ignore')

import json
import os
import argparse
from typing import Counter
from zemberek import TurkishTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from logistic_regression import LogisticRegressionModel
from sklearn.preprocessing import LabelEncoder


def parse_data(files, args): # parse the data and return a dictionary
    data = {}
    pbar = tqdm(files)
    for f_name in pbar:
        pbar.set_description(f'parsing: {f_name}')
        with open('data/' + args.raw_data_dir + '/' + f_name, encoding='utf-8') as f:
            f_number = f_name.split('.')[0]
            data[f_number] = json.load(f)['ictihat']

    return data


def export_data(data, args): # clean the data and export it
    tokenizer = TurkishTokenizer.DEFAULT

    pbar = tqdm(data)
    for key in pbar:
        pbar.set_description(f'cleaning file {key}')
        tokens = tokenizer.tokenize(data[key])

        filtered_tokens = []
        for token in tokens:
            if token.type_.name != 'Word': continue # remove anything which is not a word
            if len(token.content) == 1: continue # remove anything which is a single character
            filtered_tokens.append(token.content.lower())

        data[key] = ' '.join(filtered_tokens)

    with open(f'data/{args.f_name}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=4)


def split_dataset(training_ratio=0.8, test_ratio=0.2):
    with open('data/labels.json', encoding='utf-8') as f:
        labels = json.load(f)

    X = list(labels.keys())
    y = list(labels.values())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42, stratify=y)    

    splitted_data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
    with open('data/splitted_data.json', 'w', encoding='utf-8') as f:
        json.dump(splitted_data, f, ensure_ascii=False, sort_keys=True, indent=4)


def read_data(filenames, data):
    X = []
    for f_name in filenames:
        X.append(data[f_name])
    
    return X


def main(args):
    if args.clean_data:
        data = parse_data(os.listdir('data/' + args.raw_data_dir), args)
        export_data(data, args)

    elif args.split_data:
        split_dataset()

    else:
        with open(f'data/{args.f_name}.json', encoding='utf-8') as f:
            data = json.load(f)

        with open(f'data/splitted_data.json', 'r', encoding='utf-8') as f:
            splitted_data = json.load(f)
        
        X_train = read_data(splitted_data['X_train'], data)
        X_test = read_data(splitted_data['X_test'], data)

        le = LabelEncoder()
        le.fit(splitted_data['y_train'])
        y_train = le.transform(splitted_data['y_train'])
        y_test = le.transform(splitted_data['y_test'])

        if args.model == 'logistic_regression':
            model = LogisticRegressionModel(X_train, y_train, X_test, y_test)
            model.train(hyperparam_tuning=args.hyperparam_tuning)
            model.predict()
            model.confusion_matrix(str_labels=list(le.classes_))
            model.classification_report(str_labels=list(le.classes_))


def parse_args():
    parser = argparse.ArgumentParser("Crime Classification")
    parser.add_argument('--clean_data', type=bool, default=False, help='clean the raw data')
    parser.add_argument('--raw_data_dir', type=str, default='2021-01', help='raw dataset directory')
    parser.add_argument('--f_name', type=str, default='dataset', help='file name')
    parser.add_argument('--split_data', type=bool, default=False, help='split the dataset (training, test)')
    parser.add_argument('--model', type=str, default='logistic_regression', help='learning model')
    parser.add_argument('--hyperparam_tuning', type=bool, default=False, help='hyperparameter tuning')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
