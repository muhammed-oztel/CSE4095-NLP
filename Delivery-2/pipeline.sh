#!/bin/bash

# clean and tokenize the dataset
python main.py --clean_data True --raw_data_dir 2021-01 --f_name dataset

# split data into training and test sets
python main.py --split_data True

# perform logistic regression
python main.py --model logistic_regression
# evaluate logistic regression model on test set
python main.py --model logistic_regression --test_model True

# perform svm
python main.py --model svm
# evaluate svm model on test set
python main.py --model svm --test_model True

# train multinomial naive bayes model
python main.py --model multi_naive_bayes
# evaluate multinomial naive bayes on test set
python main.py --model multi_naive_bayes --test_model True

# perform random forest
python main.py --model random_forest
# evaluate random forest model on test set
python main.py --model random_forest --test_model True

# perform ada_boost
python main.py --model ada_boost
# evaluate ada_boost model on test set
python main.py --model ada_boost --test_model True

# perform mvoting
python main.py --model mvoting
# evaluate mvoting model on test set
python main.py --model mvoting --test_model True