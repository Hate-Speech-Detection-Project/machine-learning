
class Article:
    XPATH_ARTICLE_HEADING = 'header[data-ct-area="articleheader"]'
    XPATH_RESSORT = '//*[@id="navigation"]/nav[3]/ul/li[1]/a/span'
    #XPATH_ARTICLE_BODY = '//*[@id="js-article"]/div[1]/section/p/text()'
    XPATH_ARTICLE_BODY = '//*[@id="js-article"]/div[1]/*//p//text()'
    XPATH_ARTICLE_HEAD = '/html/head'

    def __init__(self):
        self.id = ""
        self.text_body = ""
        self.heading = ""
        self.ressort = ""
        self.url = ""

    def set_url(self, url):
        self.url = url

    def get_url(self):
        return self.url

    def set_id(self,id):
        self.id = id

    def get_id(self):
        return self.id

    def set_heading(self, heading):
        self.heading = heading

    def set_body(self, body):
        self.text_body = body

    def set_ressort(self, ressort):
        self.ressort = ressort

    def get_body(self):
        return self.text_body

    def get_heading(self):
        return self.heading

    def get_ressort(self):
        return self.ressort
