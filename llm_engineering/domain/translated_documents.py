from abc import ABC
from typing import Optional

from pydantic import UUID4

from .base import VectorBaseDocument
from .types import DataCategory


class TranslatedDocument(VectorBaseDocument, ABC):
    content: str
    platform: str


class TranslatedTranscriptionDocument(TranslatedDocument):
    name: str
    filepath: str

    class Config:
        name = "translated_transcription"
        category = DataCategory.MEETING_TRANSCRIPTION
        use_vector_index = False


class TranslatedMLBookDocument(TranslatedDocument):
    filepath: str
    name: str
    author: str

    class Config:
        name = "translated_ml_book"
        category = DataCategory.ML_BOOK
        use_vector_index = False