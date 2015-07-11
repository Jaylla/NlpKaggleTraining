import pandas as pd
import nltk.data
import logging
import numpy as np
from sklearn.cluster import KMeans
import time
from sklearn.ensemble import RandomForestClassifier
from TextPreparing import review_to_sentences, review_to_wordlist
from gensim.models import word2vec


# I need this func in order to avoid
# OverflowError: Python int too large to convert to C long
def hash32(value):
    return hash(value) & 0xffffffff


def make_feature_vec(words, word2vec_model, num_features):
    # Function to average all of the word vectors in a given paragraph
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
    for word in words:
        if word in index2word_set:
            nwords += 1.
            feature_vec = np.add(feature_vec, word2vec_model[word])
    #
    # Divide the result by the number of words to get the average
    feature_vec = np.divide(feature_vec, nwords)
    return feature_vec


def get_avg_reviews_array(reviews, word2vec_model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    c = 0.
    # Preallocate a 2D numpy array, for speed
    review_feature_vecs = np.zeros((len(reviews), num_features), dtype="float32")

    for rev in reviews:
        # makes average feature vectors
        review_feature_vecs[c] = make_feature_vec(rev, word2vec_model, num_features)
        #
        c += 1.

    return review_feature_vecs


def create_bag_of_centroids(wordlist, word_to_centroid):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_to_centroid.values()) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_to_centroid:
            index = word_to_centroid[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


def create_model():
    # Set values for various parameters
    features_count = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    #
    print("Training model...")
    w2v_model = word2vec.Word2Vec(sentences, workers=num_workers, size=features_count,
                                  min_count=min_word_count,
                                  window=context, sample=downsampling, hashfxn=hash32)
    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    w2v_model.init_sims(replace=True)
    model_name = "300features_40minwords_10context"
    w2v_model.save(model_name)
    return w2v_model


def prepare_rewiews(reviews):
    clean_reviews = []
    for r in reviews:
        clean_reviews.append(review_to_wordlist(r, remove_stopwords=True))

    return clean_reviews


def make_prediction_for_avg_vecs(train_reviews, test_reviews):
    #
    features_count = 300
    #
    train_data_vecs = get_avg_reviews_array(train_reviews, model, features_count)
    test_data_vecs = get_avg_reviews_array(test_reviews, model, features_count)
    #
    # **************************************************************
    # Fit a random forest to the training data, using 100 trees
    forest_classifier = RandomForestClassifier(n_estimators=100)
    forest_classifier = forest_classifier.fit(train_data_vecs, train["sentiment"])
    # Test & extract results
    #
    res = forest_classifier.predict(test_data_vecs)
    # Write the test results
    out_file = pd.DataFrame(data={"id": test["id"], "sentiment": res})
    out_file.to_csv("F:\Data Mining\word2vec-nlp-tutorial\Data\Word2Vec_AverageVectors_Base.csv",
                    index=False, quoting=3)


# Read data from files
train = pd.read_csv("F:\Data Mining\word2vec-nlp-tutorial\Data\labeledTrainData.tsv",
                    header=0, delimiter="\t", quoting=3)
test = pd.read_csv("F:\Data Mining\word2vec-nlp-tutorial\Data\\testData.tsv",
                   header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("F:\Data Mining\word2vec-nlp-tutorial\Data\\unlabeledTrainData.tsv",
                              header=0, delimiter="\t", quoting=3)

# Load the punkt tokenizer
punkt_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences = []  # Initialize an empty list of sentences

print("Parsing sentences from training set")
for review in train["review"]:
    sentences += review_to_sentences(review, punkt_tokenizer)

print("Parsing sentences from unlabeled set")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, punkt_tokenizer)

# creates nice output messages
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# train w2v model
model = create_model()

# ***************** Clusterize words *************************
start = time.time()

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.syn0
num_clusters = word_vectors.shape[0] // 5

# Initialize a k-means object and use it to extract centroids
kmeans_clustering = KMeans(n_clusters=num_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print("Time taken for K Means clustering: ", elapsed, "seconds.")

# Create a Word / Index dictionary, mapping each vocabulary word to
# a cluster number
word_centroid_map = dict(zip(model.index2word, idx))

# *****************************************************************

clean_train_reviews = prepare_rewiews(train["review"])
clean_test_reviews = prepare_rewiews(test["review"])

# ****************************************************************
# make_prediction_for_avg_vecs(clean_train_reviews, clean_test_reviews)
# ****************************************************************

# Pre-allocate an array for the training set bags of centroids (for speed)
train_centroids = np.zeros((train["review"].size, num_clusters), dtype="float32")

# Transform the training set reviews into bags of centroids
counter = 0
for review in clean_train_reviews:
    train_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1

# Repeat for test reviews
test_centroids = np.zeros((test["review"].size, num_clusters), dtype="float32")

counter = 0
for review in clean_test_reviews:
    test_centroids[counter] = create_bag_of_centroids(review, word_centroid_map)
    counter += 1

# *******************************************************************
# Fit a random forest and extract predictions
forest = RandomForestClassifier(n_estimators=100)

# Fitting the forest may take a few minutes
print("Fitting a random forest to labeled training data...")
forest = forest.fit(train_centroids, train["sentiment"])
result = forest.predict(test_centroids)

# Write the test results
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("F:\Data Mining\word2vec-nlp-tutorial\Data\Base_BagOfCentroids.csv", index=False, quoting=3)
