import gensim
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import LinearRegression
from TextPreparing import review_to_words, review_to_wordlist
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


# I need this func in order to avoid
# OverflowError: Python int too large to convert to C long
def hash32(value):
    return hash(value) & 0xffffffff

def make_feature_vec(words, word2vec_model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    feature_vec = np.zeros((num_features,), dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(word2vec_model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords += 1.
            feature_vec = np.add(feature_vec, word2vec_model[word])
    #
    # Divide the result by the number of words to get the average
    if nwords != 0:
        feature_vec = np.divide(feature_vec, nwords)
    return feature_vec

def get_avg_feature_vecs(reviews, word2vec_model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array

    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")

    for rev in reviews:
        #
        # Print a status message every 1000th review
        if counter % 1000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))
        #
        # makes average feature vectors
        review_feature_vecs[counter] = make_feature_vec(rev, word2vec_model, num_features)

        counter += 1.
    return review_feature_vecs

def clean_reviews(raw_data):
    # Create an empty list and append the clean reviews one by one
    reviews_count = len(raw_data["review"])
    clean_reviews = []

    print("Cleaning and parsing the movie reviews...\n")
    for i in range(0, reviews_count):
        clean_reviews.append(review_to_words(raw_data["review"][i]))

    return clean_reviews


# ************* Vectorize by TfidfVectorizer ******************************
train = pd.read_csv("F:\Data Mining\word2vec-nlp-tutorial\Data\labeledTrainData.tsv", header=0, delimiter="\t",
                    quoting=3)
test = pd.read_csv("F:\Data Mining\word2vec-nlp-tutorial\Data\\testData.tsv",
                   header=0, delimiter="\t", quoting=3)

clean_train_reviews = clean_reviews(train)

count_vectorizer = TfidfVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             max_features=20000)

# Learn the vocabulary dictionary and return term-document matrix
train_data_features = count_vectorizer.fit_transform(clean_train_reviews)


# ************* Select best features ******************************
select = SelectKBest(chi2, k=500)
X_new = select.fit_transform(train_data_features, train["sentiment"])

names = count_vectorizer.get_feature_names()

# get_support - boolean array of shape [# input features],
# in which an element is True iff its corresponding feature is selected for retention
selected_words = np.asarray(names)[select.get_support()]
# print(', '.join(selected_words))


# ************* Make average vectors of w2v representation of top 1000 words ***************************
model = gensim.models.Word2Vec.load("300features_40minwords_10context")
features_count = 300

train_reviews = []
for review in train["review"]:
    train_reviews.append(review_to_wordlist(review, vocabulary=selected_words))

trainDataVecs = get_avg_feature_vecs(train_reviews, model, features_count)

print("Creating average feature vecs for test reviews")
test_reviews = []
for review in test["review"]:
    test_reviews.append(review_to_wordlist(review, vocabulary=selected_words))

testDataVecs = get_avg_feature_vecs(test_reviews, model, features_count)


# ************* Make a prediction ******************************

model = LinearRegression()

print("Fitting a random forest to labeled training data...")
model = model.fit(trainDataVecs, train["sentiment"])

# Test & extract results
result = model.predict(testDataVecs)

# Write the test results
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("F:\Data Mining\word2vec-nlp-tutorial\Data\W2V_AvgVectors_Top500_chi2_LinearRegression.csv",
              index=False, quoting=3)
