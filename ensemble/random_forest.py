from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import re
import pandas as pd
import nltk

from nltk.corpus import stopwords # Import the stop word list

class RandomForestClassifier:
    def __init__(self):
        self.model = None
        pass

    def comment_to_words(raw_comment):
        # Function to convert a raw review to a string of words
        # The input is a single string (a raw movie review), and
        # the output is a single string (a preprocessed movie review)
        #
        # 1. Remove HTML
        comment_text = BeautifulSoup(raw_comment).get_text()
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

    def createFeatureMatrix(self, df):
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

            # Initialize the "CountVectorizer" object, which is scikit-learn's
            # bag of words tool.
            vectorizer = CountVectorizer(analyzer="word",
                                         tokenizer=None,
                                         preprocessor=None,
                                         stop_words=None,
                                         max_features=10000)

            # fit_transform() does two functions: First, it fits the model
            # and learns the vocabulary; second, it transforms our training data
            # into feature vectors. The input to fit_transform should be a list of
            # strings.
            comment_data_features = vectorizer.fit_transform(clean_train_comments)

            # Numpy arrays are easy to work with, so convert the result to an
            # array
            comment_data_features = comment_data_features.toarray()

        return comment_data_features

    def fit(self, train_df):
        print("Training the random forest...")

        # Initialize a Random Forest classifier with 100 trees
        forest = RandomForestClassifier(n_estimators=100)

        # Fit the forest to the training set, using the bag of words as
        # features and the sentiment labels as the response variable
        #
        # This may take a few minutes to run
        trainingFeatures = self.createFeatureMatrix(self, train_df);
        self.model = forest.fit(trainingFeatures, train_df["hate"])
        print("done")


    def test(self, test_df):
        # Create an empty list and append the clean reviews one by one
        num_comments = len(test_df["comment"])
        clean_test_comments = []

        print("Cleaning and parsing the test set movie reviews...\n")
        for i in range(0, num_comments):
            if (i + 1) % 1000 == 0:
                print("Comment %d of %d\n" % (i + 1, num_comments))
            clean_comment = self.comment_to_words(test_df["comment"][i])
            clean_test_comments.append(clean_comment)

        vectorizer = CountVectorizer(analyzer="word",
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=10000)

        # Get a bag of words for the test set, and convert to a numpy array
        test_data_features = vectorizer.transform(clean_test_comments)
        test_data_features = test_data_features.toarray()

        # Use the random forest to make sentiment label predictions
        result = self.model.predict(test_data_features)

        # Copy the results to a pandas dataframe with an "id" column and
        # a "sentiment" column
        output = pd.DataFrame(data={"cid": test_df["cid"], "hate": result})

        verification = pd.DataFrame(data={"cid": test_df["cid"], "hate": test_df["hate"]})
        merged = pd.merge(verification, output, on="cid")

        equal = merged.loc[merged["hate_x"] == merged["hate_y"]]
        acc = equal["cid"].count() / verification["cid"].count()

        print("Accuracy", acc)
        return acc

    def predict(self, comment):
        df = pd.Series([comment])
        features = self.createFeatureMatrix(self, df)

        return self.model.predict(features)