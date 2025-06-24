
from steps import feature_engineering as fe_steps
from llm_engineering.domain.documents import MLBookDocument, TranscriptionDocument, GithubCodeDocument


def feature_engineering() -> list[str]:
    raw_ml_book_documents = MLBookDocument.bulk_find()
    #raw_ml_book_documents_names = list(map(lambda x: x.name, raw_ml_book_documents))
    raw_ml_book_documents_names = []

    raw_transcription_documents = TranscriptionDocument.bulk_find()
    raw_transcription_documents_names = list(map(lambda x: x.name, raw_transcription_documents))

    raw_github_code_documents = GithubCodeDocument.bulk_find()
    # raw_github_code_documents_names = list(map(lambda x: x.name, raw_github_code_documents))
    raw_github_code_documents_names=[]

    raw_documents = fe_steps.query_data_warehouse(raw_ml_book_documents_names,
                                                  raw_transcription_documents_names,
                                                  raw_github_code_documents_names)

    # translate documents
    # NOTE: Only needed for the Transcriptions
    # translated_documents = fe_steps.translate_documents(raw_documents)

    # clean documents
    cleaned_documents = fe_steps.clean_documents(raw_documents)

    # summary documents
    summary_documents = fe_steps.summary_documents(cleaned_documents)

    last_step_1 = fe_steps.load_to_vector_db(summary_documents)

    # get embedded documents
    embedded_documents = fe_steps.chunk_and_embed(summary_documents)
    last_step_2 = fe_steps.load_to_vector_db(embedded_documents)

feature_engineering()