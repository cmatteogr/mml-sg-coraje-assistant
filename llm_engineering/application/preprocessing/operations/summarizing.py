from langchain_google_genai import ChatGoogleGenerativeAI
from llm_engineering.application.rag.prompt_templates import SummarizeMLTranscriptionTemplate, \
    SummarizeMLTranscriptionGetMetadataTemplate, SummarizeMLBookIndexTemplate
import json


def summarize_transcription_text(transcription: str, llm_model: str = "gemini-2.0-flash") -> str:
    """
    Summarize using LLM
    :param transcription: Transcription to summarize
    :param llm_model: LLM model to use
    :return: Text summarized
    """
    # init prompt for ML Transcription summaries

    prompt = SummarizeMLTranscriptionTemplate().create_template()
    model = ChatGoogleGenerativeAI(model=llm_model)

    chain = prompt | model
    # execute summary
    response = chain.invoke({"transcription": transcription})
    result = response.content
    # return summary
    return str(result)


def summarize_transcription_extract_metadata_text(transcription: str, llm_model: str = "gemini-2.0-flash") -> dict:
    """
    Summarize metadata using Google Cloud Translation API.
    :param transcription: Transcription to summarize
    :param llm_model: Google LLM model to use
    :return: dict with metadata
    """
    # init prompt for ML Transcription summaries

    prompt = SummarizeMLTranscriptionGetMetadataTemplate().create_template()
    model = ChatGoogleGenerativeAI(model=llm_model)

    chain = prompt | model
    # execute summary
    response = chain.invoke({"transcription": transcription})
    result = response.content
    result = result.replace('```json', '').replace('```', '')
    # validate json format and keys
    transcription_metadata = json.loads(str(result))
    if 'topics' not in transcription_metadata.keys():
        raise Exception("Topics not found in transcription metadata")
    if 'tools' not in transcription_metadata.keys():
        raise Exception("Tools not found in transcription metadata")
    if not transcription_metadata['topics']:
        raise Exception("Topics list is empty in transcription metadata")
    if not transcription_metadata['tools']:
        raise Exception("Tools list is empty in transcription metadata")

    # return summary metadata
    return transcription_metadata


def summarize_ml_book_home_pages_text(ml_book_home_pages: str, llm_model: str = "gemini-2.0-flash") -> dict:
    """
    Extract metadata ML Books using Google Cloud Translation API.
    :param ml_book_home_pages: ML Book home pages
    :param llm_model: Google LLM model to use
    :return: dict with metadata
    """
    # init prompt for ML book

    prompt = SummarizeMLBookIndexTemplate().create_template()
    model = ChatGoogleGenerativeAI(model=llm_model)

    chain = prompt | model
    # execute summary
    response = chain.invoke({"ml_book_home_pages": ml_book_home_pages})
    result = response.content
    result = result.replace('```json', '').replace('```', '')
    # validate json format and keys
    transcription_metadata = json.loads(str(result))
    if 'authors' not in transcription_metadata.keys():
        raise Exception("Authors not found in transcription metadata")
    if 'topics' not in transcription_metadata.keys():
        raise Exception("Tools not found in transcription metadata")
    if not transcription_metadata['authors']:
        raise Exception("Authors list is empty in transcription metadata")
    if not transcription_metadata['topics']:
        raise Exception("Tools list is empty in transcription metadata")

    # return summary metadata
    return transcription_metadata
