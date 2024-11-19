import pandas as pd

from pandas import DataFrame
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from analysis.preprocessing import remove_repeated_pointers, preprocess


DATASET_PATH="/home/data/analysis/dataset_20_full.csv"

def train_decision_tree(x, y) -> DecisionTreeClassifier:
    decision_tree = DecisionTreeClassifier(random_state=52)
    decision_tree.fit(x, y)

    return decision_tree


def train_random_forest(x, y) -> RandomForestClassifier:
    random_forest = RandomForestClassifier(n_jobs=8, random_state=52)
    random_forest.fit(x, y)

    return random_forest

def test_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    print(classification_report(y_test, y_pred, digits=4))


if __name__ == "__main__":
    df = pd.DataFrame(pd.read_csv(DATASET_PATH))
    df_removed = remove_repeated_pointers(df)

    x_train, x_test, y_train, y_test = preprocess(df)
    x_train_removed, x_test_removed, y_train_removed, y_test_removed = preprocess(df_removed)

    print('Training Decision Tree...')
    dt = train_decision_tree(x_train, y_train)
    test_model(dt, x_test, y_test)
    
    print('Training Decision Tree with removed pointers...')
    dtr = train_decision_tree(x_train_removed, y_train_removed)
    test_model(dtr, x_test_removed, y_test_removed)

    print('Training Random Forest...')
    rf = train_random_forest(x_train, y_train)
    test_model(rf, x_test, y_test)
    
    print('Training Random Forest with removed pointers...')
    rfr = train_random_forest(x_train_removed, y_train_removed)
    test_model(rfr, x_test_removed, y_test_removed)

