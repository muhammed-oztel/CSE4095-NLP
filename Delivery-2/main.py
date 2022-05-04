import warnings
warnings.simplefilter('ignore')

import json
import os
import argparse
from zemberek import TurkishTokenizer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from ml_models import *
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt


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


def split_dataset(test_ratio=0.2):
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


def export_encoded_labels(le):
    if os.path.isfile('encoded_labels.json'): return
    
    with open('encoded_labels.json', 'w', encoding='utf-8') as f:
        json.dump({'labels': list(le.classes_)}, f, ensure_ascii=False, sort_keys=True, indent=4)


def read_splitted_data(args, d_type):
    with open(f'data/{args.f_name}.json', encoding='utf-8') as f:
            data = json.load(f)

    with open(f'data/splitted_data.json', 'r', encoding='utf-8') as f:
        splitted_data = json.load(f)
    
    X = read_data(splitted_data[f'X_{d_type}'], data)
    le = LabelEncoder()
    le.fit(splitted_data[f'y_{d_type}'])
    y = le.transform(splitted_data[f'y_{d_type}'])
    export_encoded_labels(le)
    return X, y, le


def vectorize(X):
    with open('vectorizer.h5', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer.transform(X)


def confusion_matrix(args, model, X_test, y_test, str_labels):
    disp = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=str_labels,
        cmap=plt.cm.Blues,
        normalize=None,
    )
    disp.ax_.set_title('Confusion Matrix')

    plt.xticks(rotation = 90, fontsize=7)
    plt.yticks(fontsize=7)
    plt.savefig(f'results/{args.model}/cm.png', dpi=300, bbox_inches='tight')


def c_report(y_test, y_pred, str_labels):
    cr = classification_report(y_test, y_pred, target_names=str_labels)
    print(cr)


def main(args):
    if args.clean_data:
        data = parse_data(os.listdir('data/' + args.raw_data_dir), args)
        export_data(data, args)

    elif args.split_data:
        split_dataset()

    elif args.test_model:
        X, y, le = read_splitted_data(args, 'test')
        X = vectorize(X)
        with open(f"results/{args.model}/{args.model}.h5", 'rb') as f:
            model = pickle.load(f)
        confusion_matrix(args, model, X, y, le.classes_)
        c_report(y, model.predict(X), le.classes_)

        with open('results/' + args.model + '/' + args.model + '.txt', 'w', encoding='utf-8') as f:
            for label in le.inverse_transform(model.predict(X)):
                f.write(label + '\n')

    else:
        X, y, _ = read_splitted_data(args, 'train')

        if args.model == 'logistic_regression':
            model = LogisticRegressionModel(X, y, args.model)
            model.train()
        
        elif args.model == 'multi_naive_bayes':
            model = MultinomialNaiveBayesModel(X, y, args.model)
            model.train()

        elif args.model == 'svm':
            model = SVMModel(X, y, args.model)
            model.train()

        elif args.model == 'random_forest':
            model = RandomForestModel(X, y, args.model)
            model.train()

        elif args.model == 'ada_boost':
            model = AdaBoostModel(X, y, args.model)
            model.train()

        elif args.model == 'mvoting':
            model = MVotingModel(X, y, args.model)
            model.train()


def parse_args():
    parser = argparse.ArgumentParser("Crime Classification")
    parser.add_argument('--clean_data', type=bool, default=False, help='clean the raw data')
    parser.add_argument('--raw_data_dir', type=str, default='2021-01', help='raw dataset directory')
    parser.add_argument('--f_name', type=str, default='dataset', help='file name')
    parser.add_argument('--split_data', type=bool, default=False, help='split the dataset (training, test)')
    parser.add_argument('--model', type=str, default='logistic_regression', help='learning model')
    parser.add_argument('--test_model', type=bool, default=False, help='evaluate model on test data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
