from lxml import html
import requests

XPATH_ARTICLE_HEADING = 'header[data-ct-area="articleheader"]'
XPATH_RESSORT = '//*[@id="navigation"]/nav[3]/ul/li[1]/a/span'
XPATH_ARTICLE_BODY = '//*[@id="js-article"]/div[1]/*//p//text()'
XPATH_ARTICLE_HEAD = '/html/head'

COMPLETE_ARTICLE_URL_SUBDOMAIN = '/komplettansicht'

class SimpleCrawler:

    def crawlArticleBody(self, url):
        page = requests.get(url + COMPLETE_ARTICLE_URL_SUBDOMAIN, headers={'referer': 'https://www.facebook.com/zeitonline/'})
        tree = html.fromstring(page.content)

        articleBody = tree.xpath(XPATH_ARTICLE_BODY)
        cleanedBody = self._filter_unnecessary_linebreaks(' '.join(articleBody))
        return cleanedBody

    def _filter_unnecessary_linebreaks(self, text):
        text = text.rstrip()
        return text.replace('\n', '').replace('\r', '')







