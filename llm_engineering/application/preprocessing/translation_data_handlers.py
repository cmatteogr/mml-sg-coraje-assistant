from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from llm_engineering.domain.translation_documents import (
    TranslatedDocument,
    TranslatedMLBookDocument,
    TranslatedTranscriptionDocument
)
from llm_engineering.domain.documents import (
    Document,
    MLBookDocument,
    TranscriptionDocument
)

from .operations.translation import translate_text

DocumentT = TypeVar("DocumentT", bound=Document)
TranslatedDocumentT = TypeVar("TranslatedDocumentT", bound=TranslatedDocument)


class TranslationDataHandler(ABC, Generic[DocumentT, TranslatedDocumentT]):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """

    @abstractmethod
    def translate(self, data_model: DocumentT) -> TranslatedDocumentT:
        pass

class MLBookTranslationHandler(TranslationDataHandler):
    def translate(self, data_model: MLBookDocument) -> TranslatedMLBookDocument:
        return TranslatedMLBookDocument(
            id=data_model.id,
            content=" #### ".join(data_model.content.values()),
            platform=data_model.platform,
            filepath=data_model.filepath,
            name=data_model.name,
            author=data_model.author,
        )

class TranscriptionTranslationHandler(TranslationDataHandler):
    def translate(self, data_model: TranscriptionDocument) -> TranslatedTranscriptionDocument:
        return TranslatedTranscriptionDocument(
            id=data_model.id,
            # content=translate_text(" #### ".join(data_model.content.values()), source_language='es', target_language='en'),
            content=" #### ".join(data_model.content.values()),
            platform=data_model.platform,
            name=data_model.name,
            filepath=data_model.filepath,
        )
