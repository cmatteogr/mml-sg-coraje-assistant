from abc import ABC
from typing import Optional

from pydantic import UUID4, Field

from .base import NoSQLBaseDocument
from .types import DataCategory


class UserDocument(NoSQLBaseDocument):
    first_name: str
    last_name: str

    class Settings:
        name = "users"

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"


class Document(NoSQLBaseDocument, ABC):
    content: dict
    platform: str

class TranscriptionDocument(Document):
    name: str
    filepath: str

    class Settings:
        name = DataCategory.MEETING_TRANSCRIPTION


class MLBookDocument(Document):
    filepath: str
    name: str
    author: str

    class Settings:
        name = DataCategory.ML_BOOK


class RepositoryDocument(Document):
    name: str
    link: str

    class Settings:
        name = DataCategory.REPOSITORIES


class PostDocument(Document):
    image: Optional[str] = None
    link: str | None = None

    class Settings:
        name = DataCategory.POSTS


class ArticleDocument(Document):
    link: str

    class Settings:
        name = DataCategory.ARTICLES
