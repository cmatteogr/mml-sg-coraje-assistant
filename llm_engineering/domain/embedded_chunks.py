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
            Content: {chunk.content}\n
            """

        return context


class EmbeddedTranscriptionChunk(EmbeddedChunk):
    name: str
    filepath: str
    topics: list[str]
    tools: list[str]

    class Config:
        name = "embedded_transcription"
        category = DataCategory.MEETING_TRANSCRIPTION
        use_vector_index = True


class EmbeddedMLBookChunk(EmbeddedChunk):
    filepath: str
    name: str
    author: list[str]
    topics: list[str]

    class Config:
        name = "embedded_ml_book"
        category = DataCategory.ML_BOOK
        use_vector_index = True


class EmbeddedGithubCodeChunk(EmbeddedChunk):
    filepath: str
    name: str
    project_path: str
    project_url: str
    sha: str
    repo: str

    class Config:
        name = "embedded_github_code"
        category = DataCategory.GITHUB_CODE
        use_vector_index = True


class EmbeddedMMLSGBaseChunk(EmbeddedChunk):
    name: str
    filepath: str

    class Config:
        name = "embedded_mml_sg_base"
        category = DataCategory.MML_SG_BASE
        use_vector_index = True
