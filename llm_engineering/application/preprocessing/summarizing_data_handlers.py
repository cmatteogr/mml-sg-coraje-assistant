from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from llm_engineering.domain.summary_documents import (
    SummaryDocument,
    SummaryMLBookDocument,
    SummaryTranscriptionDocument
)
from llm_engineering.domain.documents import (
    Document,
    MLBookDocument,
    TranscriptionDocument
)

from .operations.summarizing import summarize_transcription_text

DocumentT = TypeVar("DocumentT", bound=Document)
TranslatedDocumentT = TypeVar("SummaryDocumentT", bound=SummaryDocument)


class SummaryDataHandler(ABC, Generic[DocumentT, TranslatedDocumentT]):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """

    @abstractmethod
    def summary(self, data_model: DocumentT) -> TranslatedDocumentT:
        pass

class MLBookSummaryHandler(SummaryDataHandler):
    def summary(self, data_model: MLBookDocument) -> SummaryMLBookDocument:
        return SummaryMLBookDocument(
            id=data_model.id,
            content=data_model.content,
            platform=data_model.platform,
            filepath=data_model.filepath,
            name=data_model.name,
            author=data_model.author,
        )

class TranscriptionSummaryHandler(SummaryDataHandler):
    def summary(self, data_model: TranscriptionDocument) -> SummaryTranscriptionDocument:
        return SummaryTranscriptionDocument(
            id=data_model.id,
            content=summarize_transcription_text(transcription=data_model.content),
            platform=data_model.platform,
            name=data_model.name,
            filepath=data_model.filepath,
        )
