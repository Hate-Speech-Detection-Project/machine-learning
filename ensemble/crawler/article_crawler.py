import scrapy
import re
from w3lib.html import remove_tags, remove_tags_with_content
from scrapy.selector import Selector
from scrapy.crawler import CrawlerProcess
from crawler.article import Article

HTTP_RESPONSE_OK = 200
ID_IDENTIFIER = 'id'
URL_IDENTIFIER = 'url'
COMPLETE_ARTICLE_URL_SUBDOMAIN = '/komplettansicht'


class ArticleCrawler(scrapy.Spider):
    name = "article"
    allowed_domains = ["zeit.de"]
    handle_httpstatus_list = [404]
    urls = []
    crawled_article = None
    failed_urls = []
    process = None

    def start_requests(self):
        print('crawling....')
        count = 0
        for url in self.urls:
            yield scrapy.Request(url=url + COMPLETE_ARTICLE_URL_SUBDOMAIN,
                                 headers={'referer': 'https://www.facebook.com/zeitonline/'}, callback=self.parse,
                                 method='GET',
                                 meta={ID_IDENTIFIER: count, URL_IDENTIFIER: url[0]},
                                 )
            count += 1

    def __init__(self):
        super().__init__(self)

    def parse(self, response):
        if response.status != HTTP_RESPONSE_OK:
            self.failed_urls.append([response.meta[ID_IDENTIFIER], response.status, response.url])
        else:
            article = (self._create_article_from_response(response))
            ArticleCrawler.crawled_article = article

    @staticmethod
    def get_failed_urls():
        return ArticleCrawler.failed_urls

    @staticmethod
    def create_crawler():
        ArticleCrawler.process = CrawlerProcess({
            'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
        })
        ArticleCrawler.process.crawl(ArticleCrawler)


    @staticmethod
    def start_crawler(url):
        ArticleCrawler.urls.append(url)
        ArticleCrawler.process.start()  # the script will block here until the crawling is finished


    # creates an article based on the crawler-response
    def _create_article_from_response(self, response):
        article = Article()

        article.set_id(response.meta[ID_IDENTIFIER])
        article.set_url(response.meta[URL_IDENTIFIER])

        sel = Selector(response)

        heading = ""
        text_areas = sel.css(Article.XPATH_ARTICLE_HEADING).xpath('*//div//text()').extract()
        for t in text_areas:
            heading += t
        heading = self._filter_unnecessary_linebreaks(heading)
        article.set_heading(self._filter_text_from_markup(heading))

        ressort = response.xpath(Article.XPATH_RESSORT).extract_first()
        if ressort is not None:
            article.set_ressort(self._filter_text_from_markup(ressort).lower())
        else:
            self._parse_html_head_and_set_ressort(response, article)

        paragraphs = sel.css('div[itemprop="articleBody"]').xpath('*//p//text()').extract()

        body = ""
        for p in paragraphs:
            body += p
        body = self._filter_unnecessary_linebreaks(body)

        article.set_body(body)

        return article

    # removes markup-tags from the given text
    def _filter_text_from_markup(self, markup):
        return remove_tags(remove_tags_with_content(markup, ('script',)))

    def _filter_unnecessary_linebreaks(self, text):
        text = text.rstrip()
        return text.replace('\n', '').replace('\r', '')

    # parses the html-header in order to find ressorts in the scripts for the given article
    def _parse_html_head_and_set_ressort(self, response, article):
        header = response.xpath(Article.XPATH_ARTICLE_HEAD)[0].extract()
        # extracts all occurrences of 'ressort': "..."  or 'sub_ressort': "..." in the html-header in order
        # to get the ressort
        ressort = self._find_ressort_by_regex('\'ressort\': "(.+)"', header)
        if (ressort is None):
            ressort = self._find_ressort_by_regex('\'sub_ressort\': "(.+)"', header)

        # set the specific ressort
        article.set_ressort(ressort)

    def _find_ressort_by_regex(self, regex, text):
        ressortMatch = re.search(regex, text)
        ressort = None
        if ressortMatch is not None:
            # the string  'ressort': "politik"  is trimmed to politik
            ressort = re.search('"(.+)"', ressortMatch.group(0)).group(0).replace('"', '')
        return ressort
