import string
from collections import namedtuple

import contractions
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.utils import Bunch
from tqdm import tqdm

# Define list of stopwords, lemmatizer to use, and punctuations to ignore
stopwords_list = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
ignored = string.punctuation + "£" + "–" + "—" + "0123456789"

# function to remove contractions
def remove_contractions(sentence):
    edited = []
    for word in sentence.split():
        word = word.lower()
        edited.append(contractions.fix(word))
    return " ".join(edited)


# this is the pre-processing function applied to each review
def process_sentence(sentence):

    # 1. Remove word contractions
    sentence = remove_contractions(sentence)

    # 2. Tokenize using NLTK Word Tokenizer
    tokens = word_tokenize(sentence)

    # 3. Convert to lower case,
    #    also remove words which contain any punctuations or numbers
    words = [
        word.lower()
        for word in tokens
        if all(char not in ignored for char in word)
    ]

    # 4. Lemmatize using NLTK WordNet Lemmatizer
    #    and also exclude words which have length <= 3
    processed = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stopwords_list and len(word) > 3
    ]

    # 5. Join the processed tokens, separating them by a space.
    return " ".join(processed)


# Define a named tuple to store the processed reviews
Review = namedtuple("Review", ["sentiment", "review"])

# read the data from train_file.csv in
# and return a list of named tuples
# dev percent is the percentage of the data to be used (helpful for debugging)
def read_data(filename="../data/train_file.csv", dev_percent=0) -> list:

    # 1. Load CSV file and add in column headers
    if dev_percent:
        train_data = pd.read_csv(
            filename,
            names=["sentiment", "review"],
            nrows=int(dev_percent * 18000),
        )
    else:
        train_data = pd.read_csv(filename, names=["sentiment", "review"])

    # 2. Process Reviews, one sentence at a time
    if dev_percent:
        processed_reviews = [
            Review(row["sentiment"], process_sentence(row["review"]))
            for index, row in tqdm(
                train_data.iterrows(),
                desc="Processing Reviews ... ",
                total=int(dev_percent * 18000),
            )
        ]
    else:
        processed_reviews = [
            Review(row["sentiment"], process_sentence(row["review"]))
            for index, row in tqdm(
                train_data.iterrows(),
                desc="Processing Reviews ... ",
                total=18000,
            )
        ]

    # 3. Add a column to the dataframe that contains the processed reviews
    train_data["processed_review"] = [
        review.review for review in processed_reviews
    ]

    X = train_data["processed_review"]
    y = train_data["sentiment"]
    return X, y


def read_test_data(filename="../data/test_file.csv", dev_percent=0) -> list:

    # 1. Load CSV file and add in column headers
    if dev_percent:
        test_data = pd.read_csv(
            filename,
            names=["review"],
            nrows=int(dev_percent * 18000),
        )
    else:
        test_data = pd.read_csv(filename, names=["review"])

    # 2. Process Reviews, one sentence at a time
    if dev_percent:
        processed_reviews = [
            process_sentence(row["review"])
            for index, row in tqdm(
                test_data.iterrows(),
                desc="Processing Test Reviews ... ",
                total=int(dev_percent * 18000),
            )
        ]
    else:
        processed_reviews = [
            process_sentence(row["review"])
            for index, row in tqdm(
                test_data.iterrows(),
                desc="Processing Test Reviews ... ",
                total=18000,
            )
        ]

    # 3. Add a column to the dataframe that contains the processed reviews
    test_data["processed_review"] = list(processed_reviews)

    return test_data["processed_review"]


def ecommerce_sentiment_analysis(
    filename="../data/train_file.csv", dev_percent=0
):
    X, y = read_data(filename, dev_percent)
    # print(type(X), type(y))
    target_names = ["negative", "positive"]
    class_to_idx = dict(zip(np.unique(y), target_names))
    return Bunch(
        data=X, target=y, class_to_idx=class_to_idx, target_names=target_names
    )


def test_data(filename="../data/test_file.csv", dev_percent=0):
    X = read_test_data(filename, dev_percent)
    # create y as the same type and size as X, but all values are 0
    y = np.zeros(X.shape)

    assert X.shape == y.shape
    # print(type(X), type(y))
    assert np.all(y == 0)
    return Bunch(data=X, target=y)


# write main function to test the code
if __name__ == "__main__":
    dev_percent = 0.001
    # # Read raw data, process it and export df to csv
    dataset = ecommerce_sentiment_analysis(
        "../data/train_file.csv", dev_percent
    )
    test_dataset = test_data("../data/test_file.csv", dev_percent)
