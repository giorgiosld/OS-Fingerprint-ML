import os
import sys
import csv
import time
from datetime import datetime

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np

from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier

from dataset.preprocessing import remove_repeated_pointers, preprocess

dataset_path="/home/data/dataset/dataset_20_full.csv"
results_dir="results-bow-d20"


def train_decision_tree(x, y):
    decision_tree = DecisionTreeClassifier(random_state=52)
    decision_tree.fit(x, y)

    return decision_tree


def train_decision_tree_rsearch(x, y):
    decision_tree = DecisionTreeClassifier(random_state=52)

    param_distributions = {
        'criterion': ['gini', 'entropy'],
        'max_depth': np.arange(1, 20),
        'min_samples_split': np.arange(2, 20),
        'min_samples_leaf': np.arange(1, 20),
        'max_features': [None, 'sqrt', 'log2'],
        'class_weight': [None, 'balanced']
    }

    random_search = RandomizedSearchCV(
        estimator=decision_tree,
        param_distributions=param_distributions,
        n_iter=100,
        scoring='accuracy',
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(x, y)
    return random_search


def train_random_forest(x, y):
    random_forest = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=52)
    random_forest.fit(x, y)

    return random_forest


def train_lightgbm(x, y):
    lgbm = LGBMClassifier(n_estimators=500, n_jobs=-1, random_state=52)
    lgbm.fit(x.astype('float32'), y)

    return lgbm


def test_model(model, name, x_test, y_true):
    y_pred = model.predict(x_test.astype('float32'))
    encoder = LabelEncoder()
    y_true_enc = encoder.fit_transform(y_true)
    y_pred_enc = encoder.transform(y_pred)

    with open(f'{results_dir}/results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([name,
            model.score(x_test.astype('float32'), y_true),
            precision_score(y_true_enc, y_pred_enc, average='macro'),
            recall_score(y_true_enc, y_pred_enc, average='macro'),
            f1_score(y_true_enc, y_pred_enc, average='macro')])

    plot_confusion_matrix(y_true, y_pred, name, f'{name}_confusion_matrix.png')
    print(classification_report(y_true, y_pred, digits=4))


def plot_confusion_matrix(y_true, y_pred, model_name: str, outfile: str):
    m = confusion_matrix(y_true, y_pred)
    ax = sb.heatmap(m, annot=True, cmap='Blues')
    ax.set_title(f'{model_name} Confusion matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('True Values')
    plt.savefig(os.path.join(results_dir, outfile))
    plt.clf()


def evaluate_models(df, preprocess_method, extra_name = ''):
    x_train, x_test, y_train, y_test = preprocess(df, preprocess_method)

    print(f'Training Decision Tree{extra_name}...')
    dt = train_decision_tree(x_train, y_train)
    test_model(dt, f'DecisionTree{extra_name}', x_test, y_test)

    print(f'Training Random Forest{extra_name}...')
    rf = train_random_forest(x_train, y_train)
    test_model(rf, f'RandomForest{extra_name}', x_test, y_test)

    print(f'Training Light GBM{extra_name}...')
    rf = train_lightgbm(x_train, y_train)
    test_model(rf, f'LightGBM{extra_name}', x_test, y_test)

    #evaluate_dt_rsearch(x_train, x_test, y_train, y_test)


def evaluate_dt_rsearch(x_train, x_test, y_train, y_test):
    print('Training Decision Tree with Random Search optimization...')
    random_search = train_decision_tree_rsearch(x_train, y_train)
    print("Best Parameters:", random_search.best_params_)
    print("Best Cross-Validation Accuracy:", random_search.best_score_)

    best_model = random_search.best_estimator_
    test_model(best_model, 'DecisionTreeRSearch', x_test, y_test)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage: python training.py <results_dir_name> <preprocess_method> <dataset_size>')
        sys.exit()
    results_dir = sys.argv[1]
    print(f'Saving results in {results_dir}')

    if len(sys.argv) == 4:
        dataset_path = f"/home/data/dataset/dataset_{sys.argv[3]}_full.csv"

    print(f'Loading dataset from {dataset_path}...')
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
        with open(f'{results_dir}/results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['model_name', 'accuracy', 'precision', 'recall', 'f1_score'])

    df = pd.DataFrame(pd.read_csv(dataset_path))
    evaluate_models(df, sys.argv[2])

    df_removed = remove_repeated_pointers(df)
    evaluate_models(df_removed, sys.argv[2], '-Removed')
