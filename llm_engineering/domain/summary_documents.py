from abc import ABC

from .base import VectorBaseDocument
from .types import DataCategory


class SummaryDocument(VectorBaseDocument, ABC):
    content: str
    platform: str


class SummaryTranscriptionDocument(SummaryDocument):
    name: str
    filepath: str

    class Config:
        name = "summary_transcription"
        category = DataCategory.MEETING_TRANSCRIPTION
        use_vector_index = False


class SummaryMLBookDocument(SummaryDocument):
    filepath: str
    name: str
    author: str

    class Config:
        name = "summary_ml_book"
        category = DataCategory.ML_BOOK
        use_vector_index = False