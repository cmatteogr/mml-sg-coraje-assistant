from abc import ABC
from typing import Optional

from pydantic import UUID4

from .base import VectorBaseDocument
from .types import DataCategory


class CleanedDocument(VectorBaseDocument, ABC):
    content: str
    platform: str


class CleanedTranscriptionDocument(CleanedDocument):
    name: str
    filepath: str

    class Settings:
        name = DataCategory.MEETING_TRANSCRIPTION


class CleanedMLBookDocument(CleanedDocument):
    filepath: str
    name: str
    author: str

    class Settings:
        name = DataCategory.ML_BOOK