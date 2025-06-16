from abc import ABC, abstractmethod
from typing import Generic, TypeVar, cast

from IPython.core.release import author

from llm_engineering.application.networks import EmbeddingModelSingleton
from llm_engineering.domain.chunks import Chunk, TranscriptionChunk, MLBookChunk
from llm_engineering.domain.embedded_chunks import (
    EmbeddedChunk,
    EmbeddedMLBookChunk,
    EmbeddedTranscriptionChunk,
)
from llm_engineering.domain.queries import EmbeddedQuery, Query

ChunkT = TypeVar("ChunkT", bound=Chunk)
EmbeddedChunkT = TypeVar("EmbeddedChunkT", bound=EmbeddedChunk)

embedding_model = EmbeddingModelSingleton()


class EmbeddingDataHandler(ABC, Generic[ChunkT, EmbeddedChunkT]):
    """
    Abstract class for all embedding data handlers.
    All data transformations logic for the embedding step is done here
    """

    def embed(self, data_model: ChunkT) -> EmbeddedChunkT:
        return self.embed_batch([data_model])[0]

    def embed_batch(self, data_model: list[ChunkT]) -> list[EmbeddedChunkT]:
        embedding_model_input = [data_model.content for data_model in data_model]
        embeddings = embedding_model(embedding_model_input, to_list=True)

        embedded_chunk = [
            self.map_model(data_model, cast(list[float], embedding))
            for data_model, embedding in zip(data_model, embeddings, strict=False)
        ]

        return embedded_chunk

    @abstractmethod
    def map_model(self, data_model: ChunkT, embedding: list[float]) -> EmbeddedChunkT:
        pass


class QueryEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: Query, embedding: list[float]) -> EmbeddedQuery:
        return EmbeddedQuery(
            id=data_model.id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            content=data_model.content,
            embedding=embedding,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )


class MLBookEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: MLBookChunk, embedding: list[float]) -> EmbeddedMLBookChunk:
        return EmbeddedMLBookChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            document_id=data_model.document_id,
            filepath=data_model.filepath,
            name=data_model.name,
            author=data_model.author,
            topics=data_model.topics,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )



class TranscriptionEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: TranscriptionChunk, embedding: list[float]) -> EmbeddedTranscriptionChunk:
        return EmbeddedTranscriptionChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            document_id=data_model.document_id,
            name=data_model.name,
            filepath=data_model.filepath,
            topics=data_model.topics,
            tools=data_model.tools,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )
