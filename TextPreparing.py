from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup


def review_to_words(raw_review, need_to_lemmatize=False):
    # Function to convert a raw review to a string of words
    # optional lemmatization
    #
    meaningful_words = review_to_wordlist(raw_review)

    if need_to_lemmatize:
        wnl = WordNetLemmatizer()
        meaningful_words = [wnl.lemmatize(w) for w in meaningful_words]

    # 6. Join the words back into one string separated by space
    return " ".join(meaningful_words)


def review_to_wordlist(raw_review, remove_stopwords=False, vocabulary=None):
    # Function to convert a document to a list of words,
    # optionally remove all except vocabulary words.
    #
    # Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    #
    # Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # Convert words to lower case and split them
    words = review_text.lower().split()
    #
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    #
    # Add to list only words from vocabulary
    if vocabulary is not None:
        words = [w for w in words if w in vocabulary]

    return words


def clean_reviews(raw_data):
    # Create an empty list and append the clean reviews one by one
    reviews_count = len(raw_data["review"])
    clean_test_reviews = []

    print("Cleaning and parsing the movie reviews...\n")
    for i in range(0, reviews_count):
        clean_test_reviews.append(review_to_words(raw_data["review"][i]))

    return clean_test_reviews

# Define a function to split a review into parsed sentences
def review_to_sentences(raw_review, tokenizer, remove_stopwords=False):
    # Function to split a review into parsed sentences.
    # Returns a list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(raw_review.strip())
    #
    res_sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            res_sentences.append(review_to_wordlist(raw_sentence, remove_stopwords=remove_stopwords))
    #
    # Return the list of sentences (each sentence is a list of words)
    return res_sentences
