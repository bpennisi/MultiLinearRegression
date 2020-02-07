import numpy as np

def normalize_features(X):
    """Normalize features."""
    return (X - X.mean(axis=0)) / X.std(axis=0)

def _random_shuffle(X, y, seed=999):
    """Shuffle list of tuples, maintain tuples."""
    from numpy import random
    random.seed(seed)
    
    # pack, shuffle, unpack
    arr = list(zip(X, y))
    np.random.shuffle(arr)
    X = np.array([x[0] for x in arr])
    y = np.array([x[1] for x in arr])
    
    return X, y

def train_test_split(X, y, split=0.8):
    """Split data into training and testing sets."""
    
    X, y = _random_shuffle(X, y)
    
    m = X.shape[0]
    train_size = int(np.round(m * split, 2) + 1)
    
    train_X = X[:train_size]
    test_X = X[train_size:]
    train_y = y[:train_size]
    test_y = y[train_size:]
    
    return train_X, test_X, train_y, test_y
