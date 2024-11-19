import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

DATASET_PATH="/home/data/analysis/dataset_20_full.csv"

df = pd.DataFrame(pd.read_csv(DATASET_PATH))

def remove_consecutive_duplicates(trace):
    """Helper function to remove consecutive duplicates from a trace"""
    calls = trace.split(',')
    result = [calls[0]]
    for i in range(1, len(calls)):
        if calls[i] != calls[i - 1]:
            result.append(calls[i])
        
    return ','.join(result)

def remove_repeated_pointers(dump_df):
    """Remove consecutive repeated API calls within each trace (e.g., '1,1,2,3,3,1,1,4' becomes '1,2,3,1,4')"""
    dump_df['pointer_graph_content'] = dump_df['pointer_graph_content'].apply(remove_consecutive_duplicates)
    return dump_df

def vectorize_pointers(df):
    """Convert pointers into feature vectors using Bag of Words"""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['pointer_graph_content'])
    return X
    #return X.toarray()


def split_data(df, X, test_size=0.2, random_state=42):
    """Splits the data into training and test set"""
    y = df['label'].tolist()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess(df):
    """Helper function to Vectorize pointers and split dataset into train and test"""
    X = vectorize_pointers(df)
    return split_data(df, X)


