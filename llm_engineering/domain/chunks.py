from abc import ABC
from typing import Optional

from pydantic import UUID4, Field

from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.types import DataCategory


class Chunk(VectorBaseDocument, ABC):
    content: str
    platform: str
    document_id: UUID4
    metadata: dict = Field(default_factory=dict)


class TranscriptionChunk(Chunk):
    name: str
    filepath: str

    class Config:
        category = DataCategory.MEETING_TRANSCRIPTION


class MLBookChunk(Chunk):
    filepath: str
    name: str
    author: str

    class Config:
        category = DataCategory.ML_BOOK
