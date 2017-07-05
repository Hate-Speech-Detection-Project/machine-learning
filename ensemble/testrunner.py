from bag_of_words import BagOfWordsClassifier
from text_features import TextFeatureClassifier
from word2vec import Word2VecClassifier
from doc2vec import Doc2VecClassifier
from word2vec_adding import Word2VecAddingClassifier
from word2vec_ensemble import Word2VecEnsembleClassifier
from word2vec_deep_inverse_regression import Word2VecDeepInverseRegressionClassifier
import pandas as pd
import numpy as np
from beautifultable import BeautifulTable
from sklearn.metrics import confusion_matrix
import os

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)
terminal_rows, terminal_columns = os.popen('stty size', 'r').read().split()

enabled_Word2VecClassifier = True
enable_BagOfWords = False
enable_TextFeatures = False

class Predictor:
  def initialize(self, set):

    self.train_df = pd.read_csv('../../data/datasets/' + set + '/train.csv', sep=',')
    self.test_df = pd.read_csv('../../data/datasets/' + set + '/test.csv', sep=',')

    # self.bag_of_words_classifier = BagOfWordsClassifier()
    # self.bag_of_words_classifier.fit(self.train_df)
    # self.text_features_classifier = TextFeatureClassifier()
    # self.text_features_classifier.fit(self.train_df)

    if enabled_Word2VecClassifier:
        self.word2vec_classifier = Word2VecEnsembleClassifier()
        self.word2vec_classifier.fit(self.train_df)


  def result(self):
    # bow_result = self.bag_of_words_classifier.predict(self.test_df)
    # tf_result = np.round(self.text_features_classifier.test(self.test_df))
    # print(self.word2vec_classifier.test(self.test_df))
    

    y_str = self.test_df['hate']
    try:
        y_int = ((self.test_df['hate'] == 'True').astype(float))
    except:
        y_int = ((self.test_df['hate']).astype(float))

    table = BeautifulTable(max_width=min(150, int(terminal_columns)))
    table.column_headers = ["Classifier", "Accuracy", "Prec.", "Recall", "TP (True Hate)", "FP (Wrong Alarm)", "FN (Missed Hate)", "TN (Correctly discarded)"]

    # acc, tp, fp, fn, tn, prec, rec = self.calculateRow(bow_result, y_str, True)
    # table.append_row(["Bag of Words", acc, prec, rec, tp, fp, fn, tn])

    # acc, tp, fp, fn, tn, prec, rec = self.calculateRow(tf_result, y_int, True)
    # table.append_row(["Text Features", acc, prec, rec, tp, fp, fn, tn])

    if enabled_Word2VecClassifier:
        word2VecAddingClassifier_result = np.round(self.word2vec_classifier.test(self.test_df) + 0.3)
        acc, tp, fp, fn, tn, prec, rec = self.calculateRow(word2VecAddingClassifier_result, y_int, True)
        table.append_row(["Word2VecAddingClassifier", acc, prec, rec, tp, fp, fn, tn])

    print(table)

    # print("Unioned True Hate:", 
    #   len(np.union1d(
    #     (np.intersect1d(np.argwhere(bow_result == True), np.argwhere(y_str == True))),
    #     (np.intersect1d(np.argwhere(word2VecAddingClassifier_result == True), np.argwhere(y_int == True)))
    #     )))

    # print('Wrongly detected comments:')
    # print(
    #     self.test_df.iloc[(np.intersect1d(np.argwhere(bow_result == False), np.argwhere(y_str == False)))]['comment'],
    #     self.test_df.iloc[(np.intersect1d(np.argwhere(word2VecAddingClassifier_result == False), np.argwhere(y_int == False)))]['comment']
    #     )


  def calculateRow(self, predicted, real, positive_character):
    acc = np.mean(predicted == real)
    tp = self.true_positive(predicted, real, positive_character)
    fp = self.false_positive(predicted, real, positive_character)
    fn = self.false_negative(predicted, real, positive_character)
    tn = self.true_negative(predicted, real, positive_character)
    prec = (tp/(tp+fp)) if (tp+fp) > 0 else 'NaN'
    rec = (tp/(tp+fn)) if (tp+fn) > 0 else 'NaN'
    return acc, tp, fp, fn, tn, prec, rec

  def true_positive(self, predicted, real, positive_character):
    return len(np.intersect1d(np.argwhere(predicted == positive_character), np.argwhere(real == positive_character)))

  def false_positive(self, predicted, real, positive_character):
    return len(np.intersect1d(np.argwhere(predicted == positive_character), np.argwhere(real != positive_character)))

  def false_negative(self, predicted, real, positive_character):
    return len(np.intersect1d(np.argwhere(predicted != positive_character), np.argwhere(real == positive_character)))

  def true_negative(self, predicted, real, positive_character):
    return len(np.intersect1d(np.argwhere(predicted != positive_character), np.argwhere(real != positive_character)))



predictor = Predictor()
datasets = ['tiny']  # '1000', '10000', 'stratified_10000']

for set in datasets:
  print("\nDataset:", set)
  predictor.initialize(set)
  predictor.result()
