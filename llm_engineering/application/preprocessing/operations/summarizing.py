from langchain_google_genai import ChatGoogleGenerativeAI
from llm_engineering.application.rag.prompt_templates import SummarizeMLTranscriptionTemplate
import os
import getpass

def summarize_transcription_text(transcription: str, llm_model: str = "gemini-2.0-flash") -> str:
    """
    Translate text from one language to another using Google Cloud Translation API.
    :param transcription: Transcription to summarize
    :param llm_model: Google LLM model to use
    :return: Text summarized
    """
    # init prompt for ML Transcription summaries

    prompt = SummarizeMLTranscriptionTemplate().create_template()
    model = ChatGoogleGenerativeAI(model=llm_model, project='corajemml-sg')

    chain = prompt | model
    # execute summary
    response = chain.invoke({"transcription": transcription})
    result = response.content
    # return summary
    return str(result)
