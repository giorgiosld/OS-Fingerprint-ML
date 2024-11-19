import os
import csv
import time
from datetime import datetime

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import lightgbm as lgb

from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

from dataset.preprocessing import remove_repeated_pointers, preprocess

DATASET_PATH="/home/data/analysis/dataset_200_full.csv"
RESULTS_DIR="results"


def train_decision_tree(x, y) -> DecisionTreeClassifier:
    decision_tree = DecisionTreeClassifier(random_state=52)
    decision_tree.fit(x, y)

    return decision_tree


def train_random_forest(x, y) -> RandomForestClassifier:
    random_forest = RandomForestClassifier(n_estimators=2000, n_jobs=16, random_state=52)
    random_forest.fit(x, y)

    return random_forest

def train_lightgbm(x, y) -> LGBMClassifier:
    lgbm = LGBMClassifier(n_estimators=2000, n_jobs=16, random_state=52)
    lgbm.fit(x.astype('float32'), y)

    return lgbm

def test_model(model, name, x_test, y_true):
    y_pred = model.predict(x_test.astype('float32'))
    encoder = LabelEncoder()
    y_true_enc = encoder.fit_transform(y_true)
    y_pred_enc = encoder.transform(y_pred)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f'{RESULTS_DIR}/results.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([name, model.score(x_test.astype('float32'), y_true),
            precision_score(y_true_enc, y_pred_enc, average='macro'),
            recall_score(y_true_enc, y_pred_enc, average='macro'),
            f1_score(y_true_enc, y_pred_enc, average='macro'),
            ts])

    plot_confusion_matrix(y_true, y_pred, name, f'{name}_confusion_matrix.png')
    print(classification_report(y_true, y_pred, digits=4))


def plot_confusion_matrix(y_true, y_pred, model_name: str, outfile: str):
    m = confusion_matrix(y_true, y_pred)
    ax = sb.heatmap(m, annot=True, cmap='Blues')
    ax.set_title(f'{model_name} Confusion matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('True Values')
    # ax.set_xticklabels(labels)
    # ax.set_yticklabels(labels)
    plt.savefig(os.path.join(RESULTS_DIR, outfile))
    plt.clf()


if __name__ == "__main__":
    print('Loading dataset...')
    if not os.path.isdir(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    df = pd.DataFrame(pd.read_csv(DATASET_PATH))
    df_removed = remove_repeated_pointers(df)

    x_train, x_test, y_train, y_test = preprocess(df)
    x_train_removed, x_test_removed, y_train_removed, y_test_removed = preprocess(df_removed)

    print('Training Decision Tree...')
    dt = train_decision_tree(x_train, y_train)
    test_model(dt, 'DecisionTree', x_test, y_test)

    print('Training Decision Tree with removed pointers...')
    dtr = train_decision_tree(x_train_removed, y_train_removed)
    test_model(dtr, 'DecisionTree-Removed', x_test_removed, y_test_removed)

    print('Training Random Forest...')
    rf = train_random_forest(x_train, y_train)
    test_model(rf, 'RandomForest', x_test, y_test)

    print('Training Random Forest with removed pointers...')
    rfr = train_random_forest(x_train_removed, y_train_removed)
    test_model(rfr, 'RandomForest-Removed', x_test_removed, y_test_removed)

    print('Training Light GBM...')
    rf = train_lightgbm(x_train, y_train)
    test_model(rf, 'LightGBM', x_test, y_test)

    print('Training Light GBM with removed pointers...')
    rfr = train_lightgbm(x_train_removed, y_train_removed)
    test_model(rfr, 'LightGBM-Removed', x_test_removed, y_test_removed)
