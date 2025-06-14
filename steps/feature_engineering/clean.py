from typing_extensions import Annotated

from llm_engineering.application.preprocessing import CleaningDispatcher
from llm_engineering.domain.cleaned_documents import CleanedDocument


def clean_documents(
    documents: Annotated[list, "translated_documents"],
) -> Annotated[list, "cleaned_documents"]:
    cleaned_documents = []
    for document in documents:
        cleaned_document = CleaningDispatcher.dispatch(document)
        cleaned_documents.append(cleaned_document)

    return cleaned_documents


def _get_metadata(cleaned_documents: list[CleanedDocument]) -> dict:
    metadata = {"num_documents": len(cleaned_documents)}
    for document in cleaned_documents:
        category = document.get_category()
        if category not in metadata:
            metadata[category] = {}

        metadata[category]["num_documents"] = metadata[category].get("num_documents", 0) + 1

    return metadata
