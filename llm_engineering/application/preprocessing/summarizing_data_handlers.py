from abc import ABC, abstractmethod
from typing import Generic, TypeVar
import time
from llm_engineering.domain.summary_documents import (
    SummaryDocument,
    SummaryMLBookDocument,
    SummaryTranscriptionDocument,
    SummaryGithubDocument
)
from llm_engineering.domain.documents import (
    Document,
    MLBookDocument,
    TranscriptionDocument,
    GithubCodeDocument
)

from .operations.summarizing import summarize_transcription_text, summarize_transcription_extract_metadata_text, \
    summarize_ml_book_home_pages_text

DocumentT = TypeVar("DocumentT", bound=Document)
SummaryDocumentT = TypeVar("SummaryDocumentT", bound=SummaryDocument)


class SummaryDataHandler(ABC, Generic[DocumentT, SummaryDocumentT]):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """

    @abstractmethod
    def summary(self, data_model: DocumentT) -> SummaryDocumentT:
        pass

class MLBookSummaryHandler(SummaryDataHandler):
    def summary(self, data_model: MLBookDocument) -> SummaryMLBookDocument:
        home_pages_n_characters = 17000
        ml_book_home_pages = data_model.content[:home_pages_n_characters]
        ml_book_metadata = summarize_ml_book_home_pages_text(ml_book_home_pages)
        time.sleep(20)
        return SummaryMLBookDocument(
            id=data_model.id,
            content=data_model.content,
            platform=data_model.platform,
            filepath=data_model.filepath,
            name=data_model.name,
            author=ml_book_metadata['authors'],
            topics=ml_book_metadata['topics'],
        )

class TranscriptionSummaryHandler(SummaryDataHandler):
    def summary(self, data_model: TranscriptionDocument) -> SummaryTranscriptionDocument:
        # get transcription metadata
        transcription_metadata = summarize_transcription_extract_metadata_text(transcription=data_model.content)
        time.sleep(20)
        # get transcription summary transcription
        transcription_summary = summarize_transcription_text(transcription=data_model.content)
        time.sleep(20)
        return SummaryTranscriptionDocument(
            id=data_model.id,
            content=transcription_summary,
            platform=data_model.platform,
            name=data_model.name,
            filepath=data_model.filepath,
            topics=transcription_metadata['topics'],
            tools=transcription_metadata['tools']
        )

class GithubCodeSummaryHandler(SummaryDataHandler):
    def summary(self, data_model: GithubCodeDocument) -> SummaryGithubDocument:
        return SummaryGithubDocument(
            id=data_model.id,
            content=data_model.content,
            platform=data_model.platform,
            filepath=data_model.filepath,
            name=data_model.name,
            project_path=data_model.project_path,
            project_url=data_model.project_url,
            sha=data_model.sha,
            repo=data_model.repo
        )
