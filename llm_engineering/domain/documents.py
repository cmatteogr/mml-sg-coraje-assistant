from abc import ABC
from .base import NoSQLBaseDocument
from .types import DataCategory


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