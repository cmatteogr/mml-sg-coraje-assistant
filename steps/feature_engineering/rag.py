from typing_extensions import Annotated

from llm_engineering.application import utils
from llm_engineering.application.preprocessing import ChunkingDispatcher, EmbeddingDispatcher
from llm_engineering.domain.chunks import Chunk
from llm_engineering.domain.embedded_chunks import EmbeddedChunk


def chunk_and_embed(
    cleaned_documents: Annotated[list, "cleaned_documents"],
) -> Annotated[list, "embedded_documents"]:
    metadata = {"chunking": {}, "embedding": {}, "num_documents": len(cleaned_documents)}

    embedded_chunks = []
    for document in cleaned_documents:
        chunks = ChunkingDispatcher.dispatch(document)
        metadata["chunking"] = _add_chunks_metadata(chunks, metadata["chunking"])

        for batched_chunks in utils.misc.batch(chunks, 10):
            batched_embedded_chunks = EmbeddingDispatcher.dispatch(batched_chunks)
            embedded_chunks.extend(batched_embedded_chunks)

    metadata["embedding"] = _add_embeddings_metadata(embedded_chunks, metadata["embedding"])
    metadata["num_chunks"] = len(embedded_chunks)
    metadata["num_embedded_chunks"] = len(embedded_chunks)

    return embedded_chunks


def _add_chunks_metadata(chunks: list[Chunk], metadata: dict) -> dict:
    for chunk in chunks:
        category = chunk.get_category()
        if category not in metadata:
            metadata[category] = chunk.metadata

        metadata[category]["num_chunks"] = metadata[category].get("num_chunks", 0) + 1

    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))

    return metadata


def _add_embeddings_metadata(embedded_chunks: list[EmbeddedChunk], metadata: dict) -> dict:
    for embedded_chunk in embedded_chunks:
        category = embedded_chunk.get_category()
        if category not in metadata:
            metadata[category] = embedded_chunk.metadata

    return metadata
