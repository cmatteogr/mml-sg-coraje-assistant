from abc import ABC, abstractmethod
from llm_engineering.domain.documents import NoSQLBaseDocument

class BaseCrawler(ABC):
    model: type[NoSQLBaseDocument]

    @abstractmethod
    def extract(self, link: str, **kwargs) -> None: ...

LOCAL_FOLDER = '/home/cesarealice/PycharmProjects/mml-sg-coraje-assistant/data/input'
#LOCAL_FOLDER = '../data/input'
class BaseLocalCrawler(BaseCrawler, ABC):
    def __init__(self, destination_folder: str =LOCAL_FOLDER) -> None:
        self.destination_folder = destination_folder
