from typing_extensions import Annotated
from zenml import get_step_context, step

from llm_engineering.application.preprocessing import SummaryDispatcher
from llm_engineering.domain.summary_documents import SummaryDocument


@step
def summary_documents(
    documents: Annotated[list, "raw_documents"],
) -> Annotated[list, "summary_documents"]:
    summary_documents = []
    for document in documents:
        summary_document = SummaryDispatcher.dispatch(document)
        summary_documents.append(summary_document)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="summary_documents", metadata=_get_metadata(summary_documents))

    return summary_documents


def _get_metadata(summary_documents: list[SummaryDocument]) -> dict:
    metadata = {"num_documents": len(summary_documents)}
    for document in summary_documents:
        category = document.get_category()
        if category not in metadata:
            metadata[category] = {}

        metadata[category]["num_documents"] = metadata[category].get("num_documents", 0) + 1

    return metadata
