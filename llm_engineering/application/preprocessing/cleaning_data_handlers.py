from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from llm_engineering.domain.cleaned_documents import (
    CleanedDocument,
    CleanedMLBookDocument,
    CleanedTranscriptionDocument
)
from llm_engineering.domain.documents import (
    Document,
    MLBookDocument,
    TranscriptionDocument
)

from .operations import clean_text

DocumentT = TypeVar("DocumentT", bound=Document)
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)


class CleaningDataHandler(ABC, Generic[DocumentT, CleanedDocumentT]):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """

    @abstractmethod
    def clean(self, data_model: DocumentT) -> CleanedDocumentT:
        pass

class MLBookCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: MLBookDocument) -> CleanedMLBookDocument:
        return CleanedMLBookDocument(
            id=data_model.id,
            content=clean_text(" #### ".join(data_model.content.values())),
            platform=data_model.platform,
            filepath=data_model.filepath,
            name=data_model.name,
            author=data_model.author,
        )

class TranscriptionCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: TranscriptionDocument) -> CleanedTranscriptionDocument:
        return CleanedTranscriptionDocument(
            id=data_model.id,
            content=clean_text(" #### ".join(data_model.content.values())),
            platform=data_model.platform,
            name=data_model.name,
            filepath=data_model.filepath,
        )
