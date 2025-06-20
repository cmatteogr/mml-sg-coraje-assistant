from .base import BaseCrawler
from .custom_article import CustomArticleCrawler
from .ml_book import MLBookCrawler
from .transcription import TranscriptionCrawler
from .github import GithubCodeCrawler


class CrawlerDispatcher:
    def __init__(self) -> None:
        self._crawlers = {}

    @classmethod
    def build(cls) -> "CrawlerDispatcher":
        dispatcher = cls()

        return dispatcher

    def register_transcription(self) -> "CrawlerDispatcher":
        self.register("transcription", TranscriptionCrawler)

        return self

    def register_ml_book(self) -> "CrawlerDispatcher":
        self.register("ml_books", MLBookCrawler)

        return self

    def register_github_code(self) -> "CrawlerDispatcher":
        self.register("github_code", GithubCodeCrawler)

        return self

    def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
        self._crawlers[domain] = crawler

    def get_crawler(self, domain: str) -> BaseCrawler:
        # get crawler if exist using the domain
        crawler = self._crawlers.get(domain)
        # if crawler is None use the Custom Article
        if crawler is not None:
            return crawler()
        # return custom crawler
        return CustomArticleCrawler()
