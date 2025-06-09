from abc import ABC
from .base import VectorBaseDocument
from .types import DataCategory


class CleanedDocument(VectorBaseDocument, ABC):
    content: str
    platform: str


class CleanedTranscriptionDocument(CleanedDocument):
    name: str
    filepath: str

    class Config:
        name = "cleaned_transcription"
        category = DataCategory.MEETING_TRANSCRIPTION
        use_vector_index = False


class CleanedMLBookDocument(CleanedDocument):
    filepath: str
    name: str
    author: str

    class Config:
        name = "cleaned_ml_book"
        category = DataCategory.ML_BOOK
        use_vector_index = False