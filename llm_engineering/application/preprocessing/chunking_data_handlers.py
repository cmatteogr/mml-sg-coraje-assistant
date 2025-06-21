import hashlib
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from uuid import UUID
import ast
import os
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
            "chunk_size": 500,
            "chunk_overlap": 50,
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
            "chunk_size": 500,
            "chunk_overlap": 50,
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
            "chunk_size": 500,
            "chunk_overlap": 50,
        }

    def extract_all_code_chunks(selft, source_code):
        """
        Parses a Python file and extracts all top-level code chunks,
        """
        tree = ast.parse(source_code)

        all_chunks = []
        standalone_nodes = []

        def commit_standalone_chunk():
            """Helper function to process and store a block of standalone nodes."""
            if not standalone_nodes:
                return

            # Get the first and last node to get the full source segment
            start_node = standalone_nodes[0]
            end_node = standalone_nodes[-1]

            # Using ast.get_source_segment on each and joining is an option,
            # but slicing the original source is often more reliable for blocks.
            start_line = start_node.lineno
            end_line = end_node.end_lineno

            source_lines = source_code.splitlines()
            # Slice the original source lines to get the block
            chunk_code = '\n'.join(source_lines[start_line - 1:end_line])

            all_chunks.append({
                "name": None,
                "type": "standalone",
                "code": chunk_code
            })
            standalone_nodes.clear()

        # Iterate through all top-level nodes in the file's body
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                # If we encounter a function or class, first commit any pending standalone chunk.
                commit_standalone_chunk()

                # Then, process the function or class as its own chunk.
                chunk_type = "function" if isinstance(node, ast.FunctionDef) else "class"
                all_chunks.append({
                    "name": node.name,
                    "type": chunk_type,
                    "code": ast.get_source_segment(source_code, node)
                })
            else:
                # If it's any other type of node, add it to our list of standalone nodes.
                standalone_nodes.append(node)

        # After the loop, there might be a final standalone chunk (e.g., the if __name__ block)
        commit_standalone_chunk()

        return all_chunks

    def chunk_code_text(self, segments: list[dict], script_path: str):
        segments_chunks = []
        for segment in segments:
            chunks = chunk_text(segment['code'], chunk_size=self.metadata["chunk_size"],
                                chunk_overlap=self.metadata["chunk_overlap"])
            segment_type = segment['type']
            segment_name = segment['name']
            file_header = f'File: {script_path}\n'

            for chunk in chunks:
                chunk_code = f"""```python\n{chunk}```\n"""

                match segment_type:
                    case 'class':
                        chunk_complete = f"""{file_header}\nClass: {segment_name}\n{chunk_code}"""
                    case 'function':
                        chunk_complete = f"""{file_header}\nFunction: {segment_name}\n{chunk_code}"""
                    case 'standalone':
                        chunk_complete = f"""{file_header}\n{chunk_code}"""
                    case _:
                        raise Exception(f"invalid segment type: {segment_type}")

                segments_chunks.append(chunk_complete)

        return segments_chunks

    def chunk(self, data_model: CleanedGithubCodeDocument) -> list[GithubCodeChunk]:
        data_models_list = []

        cleaned_content = data_model.content

        extension_file = os.path.splitext(data_model.project_path)[-1]
        # define chunk text strategy based on file extension
        match extension_file:
            case '.py':
                fun_class_dict = self.extract_all_code_chunks(cleaned_content)
                chunks = self.chunk_code_text(fun_class_dict, data_model.project_path)
            case '.md':
                chunks = chunk_header_chunk(cleaned_content)
            case _:
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