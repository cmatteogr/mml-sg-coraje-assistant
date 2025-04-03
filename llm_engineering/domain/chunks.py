from abc import ABC
from typing import Optional

from pydantic import UUID4, Field

from llm_engineering.domain.base import VectorBaseDocument
from llm_engineering.domain.types import DataCategory


class Chunk(VectorBaseDocument, ABC):
    content: str
    platform: str
    document_id: UUID4
    author_id: UUID4
    author_full_name: str
    metadata: dict = Field(default_factory=dict)


class TranscriptionChunk(Chunk):
    name: str
    link: str

    class Settings:
        name = DataCategory.MEETING_TRANSCRIPTION


class MLBookChunk(Chunk):
    name: str
    author: str

    class Settings:
        name = DataCategory.ML_BOOK


class PostChunk(Chunk):
    image: Optional[str] = None

    class Config:
        category = DataCategory.POSTS


class ArticleChunk(Chunk):
    link: str

    class Config:
        category = DataCategory.ARTICLES


class RepositoryChunk(Chunk):
    name: str
    link: str

    class Config:
        category = DataCategory.REPOSITORIES
