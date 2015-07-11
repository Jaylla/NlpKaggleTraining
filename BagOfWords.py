from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from TextPreparing import clean_reviews


def show_word_statistic(feature_names, feature_array):
    # Sum up the counts of each vocabulary word
    dist = np.sum(feature_array, axis=0)

    # For each, print the vocabulary word and the number of times it appears in the set
    for count, tag in sorted([(count, tag) for tag, count in zip(feature_names, dist)], reverse=True):
        print(count, tag)


def make_prediction(classifier, vectorizer, test_file_name, out_file_name):
    # Read the test data
    test = pd.read_csv(test_file_name, header=0, delimiter="\t", quoting=3)
    clean_test_reviews = clean_reviews(test)

    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    test_data_features = test_data_features.toarray()

    # Use the random forest to make sentiment label predictions
    result = classifier.predict(test_data_features)

    # Copy the results to a pandas dataframe with an "id" column and
    # a "sentiment" column
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

    # Use pandas to write the comma-separated output file
    output.to_csv(out_file_name, index=False, quoting=3)


train = pd.read_csv("F:\Data Mining\word2vec-nlp-tutorial\Data\labeledTrainData.tsv", header=0, delimiter="\t",
                    quoting=3)

clean_train_reviews = clean_reviews(train)

count_vectorizer = TfidfVectorizer(analyzer="word",
                                   tokenizer=None,
                                   preprocessor=None,
                                   stop_words=None,
                                   max_features=5000)

train_data_features = count_vectorizer.fit_transform(clean_train_reviews)
# Numpy arrays are easy to work with, so convert the result to an array
train_data_features = train_data_features.toarray()

# vocab = count_vectorizer.get_feature_names()
# show_word_statistic(vocab, train_data_features)


print("Training the random forest...")
forest = RandomForestClassifier(n_estimators=100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
forest = forest.fit(train_data_features, train["sentiment"])

make_prediction(forest, count_vectorizer, "F:\Data Mining\word2vec-nlp-tutorial\Data\\testData.tsv",
                "F:\Data Mining\word2vec-nlp-tutorial\Data\Bag_of_Words_Tfidf_lemmatization_4.csv")
