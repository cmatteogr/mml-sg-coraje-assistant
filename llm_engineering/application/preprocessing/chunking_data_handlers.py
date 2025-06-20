import hashlib
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID

from llm_engineering.domain.chunks import Chunk, TranscriptionChunk, MLBookChunk, GithubCodeChunk
from llm_engineering.domain.cleaned_documents import (
    CleanedDocument,
    CleanedTranscriptionDocument,
    CleanedMLBookDocument,
    CleanedGithubCodeDocument
)

from .operations import chunk_text, chunk_header_chunk

CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)
ChunkT = TypeVar("ChunkT", bound=Chunk)


class ChunkingDataHandler(ABC, Generic[CleanedDocumentT, ChunkT]):
    """
    Abstract class for all Chunking data handlers.
    All data transformations logic for the chunking step is done here
    """

    @abstractmethod
    def chunk(self, data_model: CleanedDocumentT) -> list[ChunkT]:
        pass


class MLBookChunkingHandler(ChunkingDataHandler):
    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 250,
            "chunk_overlap": 25,
        }

    def chunk(self, data_model: CleanedMLBookDocument) -> list[MLBookChunk]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_text(
            cleaned_content, chunk_size=self.metadata["chunk_size"], chunk_overlap=self.metadata["chunk_overlap"]
        )

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = MLBookChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                document_id=data_model.id,
                filepath=data_model.filepath,
                name=data_model.name,
                author=data_model.author,
                topics=data_model.topics,
                metadata=self.metadata,
            )
            data_models_list.append(model)

        return data_models_list


class TranscriptionChunkingHandler(ChunkingDataHandler):

    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 250,
            "chunk_overlap": 25,
        }

    def chunk(self, data_model: CleanedTranscriptionDocument) -> list[TranscriptionChunk]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_header_chunk(cleaned_content)

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = TranscriptionChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                document_id=data_model.id,
                name=data_model.name,
                filepath=data_model.filepath,
                topics=data_model.topics,
                tools=data_model.tools,
                metadata=self.metadata,
            )
            data_models_list.append(model)

        return data_models_list


class GithubCodeChunkingHandler(ChunkingDataHandler):

    @property
    def metadata(self) -> dict:
        return {
            "chunk_size": 250,
            "chunk_overlap": 25,
        }

    def chunk(self, data_model: CleanedGithubCodeDocument) -> list[GithubCodeChunk]:
        data_models_list = []

        cleaned_content = data_model.content
        chunks = chunk_text(
            cleaned_content, chunk_size=self.metadata["chunk_size"], chunk_overlap=self.metadata["chunk_overlap"]
        )

        for chunk in chunks:
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()
            model = GithubCodeChunk(
                id=UUID(chunk_id, version=4),
                content=chunk,
                platform=data_model.platform,
                document_id=data_model.id,
                name=data_model.name,
                filepath=data_model.filepath,
                project_path=data_model.project_path,
                project_url=data_model.project_url,
                sha=data_model.sha,
                repo=data_model.repo,
                metadata=self.metadata,
            )
            data_models_list.append(model)

        return data_models_list