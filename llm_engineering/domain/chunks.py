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
    topics: list[str]
    tools: list[str]

    class Config:
        category = DataCategory.MEETING_TRANSCRIPTION


class MLBookChunk(Chunk):
    filepath: str
    name: str
    author: str
    author: list[str]
    topics: list[str]

    class Config:
        category = DataCategory.ML_BOOK


class GithubCodeChunk(Chunk):
    filepath: str
    name: str
    project_path: str
    project_url: str
    sha: str
    repo: str

    class Config:
        category = DataCategory.GITHUB_CODE


class MMLSGBaseChunk(Chunk):
    name: str
    filepath: str

    class Config:
        category = DataCategory.MML_SG_BASE