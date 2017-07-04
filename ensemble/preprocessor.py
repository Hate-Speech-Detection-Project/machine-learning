from bs4 import BeautifulSoup
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords # Import the stop word list

class Preprocessor:
    def __init__(self):
        self.vectorizer  = CountVectorizer(analyzer="word",
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=8)

    def comment_to_words(self, raw_comment):
        # Function to convert a raw review to a string of words
        # The input is a single string (a raw movie review), and
        # the output is a single string (a preprocessed movie review)
        #
        # 1. Remove HTML
        comment_text = BeautifulSoup(raw_comment, "html.parser").get_text()
        #
        # 2. Remove non-letters
        letters_only = re.sub("[^a-zA-Z]", " ", comment_text)
        #
        # 3. Convert to lower case, split into individual words
        words = letters_only.lower().split()
        #
        # 4. In Python, searching a set is much faster than searching
        #   a list, so convert the stop words to a set
        stops = set(stopwords.words("german"))
        #
        # 5. Remove stop words
        meaningful_words = [w for w in words if not w in stops]
        #
        # 6. Join the words back into one string separated by space,
        # and return the result.
        return (" ".join(meaningful_words))
        # return comment_text

    def trainFeatureMatrix(self, df):
        # Get the number of reviews based on the dataframe column size
        num_comments = df["comment"].size

        # Initialize an empty list to hold the clean reviews
        clean_train_comments = []

        # Loop over each review; create an index i that goes from 0 to the length
        # of the movie review list
        for i in range(0, num_comments):
            # Call our function for each one, and add the result to the list of
            # clean reviews
            clean_train_comments.append(self.comment_to_words(df["comment"][i]))

        print("Creating the bag of words...\n")

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.
        comment_data_features = self.vectorizer.fit_transform(clean_train_comments)

        # Numpy arrays are easy to work with, so convert the result to an
        # array
        comment_data_features = comment_data_features.toarray()

        return comment_data_features

    def createFeatureMatrix(self, df):
        # Create an empty list and append the clean reviews one by one
        num_comments = len(df["comment"])
        clean_test_comments = []

        print("Cleaning and parsing the test set comments...\n")
        for i in range(0, num_comments):
            if (i + 1) % 1000 == 0:
                print("Comment %d of %d\n" % (i + 1, num_comments))
            clean_comment = self.comment_to_words(df["comment"][i])
            clean_test_comments.append(clean_comment)

        # Get a bag of words for the test set, and convert to a numpy array
        test_data_features = self.vectorizer.transform(clean_test_comments)
        return test_data_features

    @staticmethod
    def convertBoolStringsToNumbers(inputArray):
        return list(map((lambda x: 1 if x == 't' else 0), inputArray))
