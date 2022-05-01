from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
import pickle
import os
import json


class MultinomialNaiveBayesModel:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.vectorize()
        os.makedirs('results/multi_naive_bayes', exist_ok=True)

    def vectorize(self):
        if os.path.isfile('vectorizer.sav'):
            self.vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
            self.X = self.vectorizer.transform(self.X)
            return

        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.X)        
        filename = 'vectorizer.sav'
        pickle.dump(self.vectorizer, open(filename, 'wb'))

    def train(self):
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        old_score = 0
        res = {'results': []}
        fold = 1
        for train_index, test_index in skf.split(self.X, self.y):
            print(f'Fold {fold}')
            fold += 1
            model = MultinomialNB()
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            score = f1_score(y_test, y_pred, average='macro')
            res['results'].append(score)
            
            if score > old_score:
                self.model = model
                old_score = score
        
        with open('results/multi_naive_bayes/results.json', 'w') as f:
            json.dump(res, f, indent=4)

        filename = f'results/multi_naive_bayes/multi_naive_bayes.sav'
        pickle.dump(self.model, open(filename, 'wb'))
