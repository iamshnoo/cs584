import warnings

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from pre_process import ecommerce_sentiment_analysis, test_data

warnings.simplefilter(action="ignore", category=FutureWarning)

# create a tfidf vectorizer object and also add relevant attributes to dataset
def vectorize(dataset, min_df=3, max_df=0.95, max_features=500):
    vectorizer = TfidfVectorizer(
        min_df=min_df, max_df=max_df, max_features=max_features
    )
    embeddings = vectorizer.fit_transform(dataset.data)
    dataset.embeddings = embeddings
    dataset.vocab = vectorizer.get_feature_names()
    dataset.vocab_length = len(vectorizer.get_feature_names())
    dataset.tf_idf_non_zero_percentage = (
        embeddings.nnz / np.prod(embeddings.shape)
    ) * 100
    return vectorizer


# create a train and validation split
def train_validation_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


if __name__ == "__main__":
    dev_percent = 0.001

    # 1. Read raw data, process it and load it like an sklearn dataset
    path = "../data/train_file.csv"
    dataset = ecommerce_sentiment_analysis(path, dev_percent)

    # 2. Vectorize the dataset
    tf_idf = vectorize(dataset)

    # 3. Split the dataset into train and validation
    X_train, X_test, y_train, y_test = train_validation_split(
        dataset.embeddings, dataset.target
    )

    print(
        type(dataset.embeddings)
    )  # <class 'scipy.sparse.csr.csr_matrix'>, Compressed Sparse Row matrix
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    path = "../data/test_file.csv"
    test_dataset = test_data(path, dev_percent)
    test_tf_idf = vectorize(test_dataset)
    testX = test_dataset.embeddings
    testy = test_dataset.target
    print(testX.shape, testy.shape)
