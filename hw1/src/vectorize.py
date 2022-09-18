import warnings

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# import personal modules
from pre_process import ecommerce_sentiment_analysis

# ignore pandas future warnings as they are coming from unused modules
warnings.simplefilter(action="ignore", category=FutureWarning)

# create a tfidf vectorizer object and also add relevant attributes to dataset
# dataset embeddings stores the tfidf vectorized data
# also, we store the vocabulary and the vocbulary size
# the function returns the vectorizer object becasue we will need it later for
# the test data.
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


#
def reduce_dimensionality(dataset, n_components=100):
    svd = TruncatedSVD(n_components=n_components)
    dataset.embeddings = svd.fit_transform(dataset.embeddings)
    return svd


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
    print("TF-IDF Embeddings shape : ", dataset.embeddings.shape)

    # 2.1 Reduce dimensionality
    svd = reduce_dimensionality(dataset, 10)
    print("TF-IDF+Truncated SVD Embeddings shape : ", dataset.embeddings.shape)

    # 3. Split the dataset into train and validation
    X_train, X_test, y_train, y_test = train_validation_split(
        dataset.embeddings, dataset.target
    )

    print(
        type(dataset.embeddings)
    )  # <class 'scipy.sparse.csr.csr_matrix'>, Compressed Sparse Row matrix
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
