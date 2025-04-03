from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from typing_extensions import Annotated
from zenml import get_step_context, step
from llm_engineering.domain.base.nosql import NoSQLBaseDocument
from llm_engineering.domain.documents import Document, MLBookDocument, TranscriptionDocument


@step
def query_data_warehouse(
    ml_book_names: list[str],
    transcription_names: list[str],
) -> Annotated[list, "raw_documents"]:
    documents = []

    results = fetch_all_data(ml_book_names, transcription_names)
    user_documents = [doc for query_result in results.values() for doc in query_result]
    documents.extend(user_documents)

    step_context = get_step_context()
    step_context.add_output_metadata(output_name="raw_documents", metadata=_get_metadata(documents))

    return documents


def fetch_all_data(ml_book_names: list[str], transcription_names: list[str]) -> dict[str, list[NoSQLBaseDocument]]:
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(__fetch_ml_books, ml_book_names): "ml_books",
            executor.submit(__fetch_transcriptions, transcription_names): "transcriptions"
        }

        results = {}
        for future in as_completed(future_to_query):
            query_name = future_to_query[future]
            try:
                results[query_name] = future.result()
            except Exception:
                logger.exception(f"'{query_name}' request failed.")

                results[query_name] = []

    return results


def __fetch_ml_books(names) -> list[NoSQLBaseDocument]:
    return MLBookDocument.bulk_find(name={"$in": names})


def __fetch_transcriptions(names) -> list[NoSQLBaseDocument]:
    return TranscriptionDocument.bulk_find(name={"$in": names})



def _get_metadata(documents: list[Document]) -> dict:
    metadata = {
        "num_documents": len(documents),
    }
    for document in documents:
        collection = document.get_collection_name()
        if collection not in metadata:
            metadata[collection] = {}

        metadata[collection]["num_documents"] = metadata[collection].get("num_documents", 0) + 1

    return metadata