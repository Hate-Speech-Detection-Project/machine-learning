from crawler.db_interface import DBInterface
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from threading import Thread

SIMILAR = 0
DIFFERENT = 90

class TopicFeatures:

    def __init__(self):
        self.dbinterface = DBInterface()

    def get_cos_similarity_for_article(self, comment, article_url):
        cos_sim_in_degree = SIMILAR
        corpus = [comment.strip()]
        article_body = self.dbinterface.get_articlebody_by_url(article_url)
        if not article_body is None:

            vector = TfidfVectorizer(min_df=1)
            corpus.extend([article_body[0]])
            vector.fit(corpus)

            tfidf_comment = vector.transform([comment])
            tfidf_article = vector.transform([article_body[0]])

            cos_sim = cosine_similarity(tfidf_comment, tfidf_article)
            cos_sim_in_degree = self._cos_to_degree(cos_sim)

        return cos_sim_in_degree



    def get_cos_similarity_for_no_hate_comments_of_article(self, comment, article_url):
        cos_sim_in_degree = SIMILAR
        corpus = [comment.strip()]
        no_hate_comments = self.dbinterface.get_comments_for_article_by_type(article_url, 'f')

        if len(no_hate_comments) != 0:

            no_hate_corpus = ''
            for tuple in no_hate_comments:
                no_hate_comment = tuple[0].strip()
                if not bool(no_hate_comment.strip()):
                    no_hate_corpus += ' ' + no_hate_comment

            if not bool(no_hate_corpus.strip()):
                return SIMILAR

            corpus.extend([no_hate_corpus])
            vector = TfidfVectorizer(min_df=1)
            vector.fit(corpus)

            tfidf_comment = vector.transform([comment])
            tfidf_no_hate = vector.transform([no_hate_corpus])

            cos_sim = cosine_similarity(tfidf_comment, tfidf_no_hate)
            cos_sim_in_degree = self._cos_to_degree(cos_sim)

        return cos_sim_in_degree

    def get_cos_similarity_for_hate_comments_of_article(self, comment, article_url):
        cos_sim_in_degree = DIFFERENT
        corpus = [comment.strip()]
        hate_comments = self.dbinterface.get_comments_for_article_by_type(article_url, 't')

        if len(hate_comments) != 0:

            hate_comments_corpus = ''
            for tuple in hate_comments:
                hate_comment = tuple[0].strip()
                if not bool(hate_comment.strip()):
                    hate_comments_corpus += ' ' + hate_comment

            if not bool(hate_comments_corpus.strip()):
                return DIFFERENT

            corpus.extend([hate_comments_corpus])

            vector = TfidfVectorizer(min_df=1)
            vector.fit(corpus)

            tfidf_comment = vector.transform([comment])
            tfidf_hate_comments = vector.transform([hate_comments_corpus])

            cos_sim = cosine_similarity(tfidf_comment, tfidf_hate_comments)
            cos_sim_in_degree = self._cos_to_degree(cos_sim)

        return cos_sim_in_degree

    #
    # def vote(self, degrees):
    #     votes = ['f','f','f']
    #     if degrees[COS_SIM_NO_HATE_INDEX] >= 85:
    #         votes[COS_SIM_NO_HATE_INDEX] = 't'
    #
    #     if degrees[COS_SIM_ARTICLE_INDEX] >= 85:
    #         votes[COS_SIM_ARTICLE_INDEX] = 't'
    #
    #     if degrees[COS_SIM_HATE_INDEX] < 60:
    #         votes[COS_SIM_HATE_INDEX] = 't'
    #
    #     if(votes.count('t') >= 2):
    #         return 't'
    #
    #     return 'f'

    def _cos_to_degree(self, cos):
        if cos <= 1.0 and cos >= 0:
            return math.degrees(math.acos(cos))

        return 0
