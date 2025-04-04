from abc import ABC

from pydantic import UUID4, Field

from llm_engineering.domain.types import DataCategory

from .base import VectorBaseDocument


class EmbeddedChunk(VectorBaseDocument, ABC):
    content: str
    embedding: list[float] | None
    platform: str
    document_id: UUID4
    metadata: dict = Field(default_factory=dict)

    @classmethod
    def to_context(cls, chunks: list["EmbeddedChunk"]) -> str:
        context = ""
        for i, chunk in enumerate(chunks):
            context += f"""
            Chunk {i + 1}:
            Type: {chunk.__class__.__name__}
            Platform: {chunk.platform}
            Author: {chunk.author_full_name}
            Content: {chunk.content}\n
            """

        return context


class EmbeddedTranscriptionChunk(EmbeddedChunk):
    name: str
    filepath: str

    class Config:
        name = "embedded_transcription"
        category = DataCategory.MEETING_TRANSCRIPTION
        use_vector_index = True


class EmbeddedMLBookChunk(EmbeddedChunk):
    filepath: str
    name: str
    author: str

    class Config:
        name = "embedded_ml_book"
        category = DataCategory.ML_BOOK
        use_vector_index = True
