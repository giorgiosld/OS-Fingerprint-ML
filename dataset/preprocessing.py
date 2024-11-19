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

def vectorize_pointers_bow(df):
    """Convert pointers into feature vectors using Bag of Words"""
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['pointer_graph_content'])
    return X

def vectorize_pointers_word2vec(df):
    """Convert pointers into feature vectors using Word2Vec"""
    # Split the pointer_graph_content into lists of pointers
    sentences = df['pointer_graph_content'].apply(lambda x: x.split(',')).tolist()
    
    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    # Create feature vectors by averaging word vectors for each pointer trace
    def get_sentence_vector(sentence):
        vectors = [model.wv[word] for word in sentence if word in model.wv]
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model.vector_size)
    
    X_word2vec = np.array([get_sentence_vector(sentence) for sentence in sentences])
    return X_word2vec

def split_data(df, X, test_size=0.2, random_state=42):
    """Splits the data into training and test set"""
    y = df['label'].tolist()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def preprocess(df, method="bow"):
    """helper function to Vectorize pointers and split dataset into train and test
    Args:
        method (str): The method to use for vectorization. Options are "bow" for Bag of Words or "word2vec" for Word2Vec.
    """
    if method == "bow":
        # Bag of Words features
        X = vectorize_pointers_bow(df)
    elif method == "word2vec":
        # Word2Vec features
        X = vectorize_pointers_word2vec(df)
    else:
        raise ValueError("Invalid method. Choose either 'bow' or 'word2vec'.")
    
    return split_data(df, X)

