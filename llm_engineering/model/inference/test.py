from loguru import logger

from llm_engineering.application.rag.retriever import ContextRetriever
from llm_engineering.domain.embedded_chunks import EmbeddedChunk
from llm_engineering.model.inference.inference_aws import LLMInferenceSagemakerEndpoint
from llm_engineering.model.inference.inference_local import LLMInferenceLocal
from llm_engineering.model.inference.run import InferenceExecutor
from llm_engineering.settings import settings

if __name__ == "__main__":
    text = ""
    logger.info(f"Running inference for text: '{text}'")

    retriever = ContextRetriever(mock=False)
    documents = retriever.search(text, k=10)
    context = EmbeddedChunk.to_context(documents)

    llm = LLMInferenceLocal()
    answer = InferenceExecutor(llm, text, context=context).execute()

    logger.info(f"Answer: '{answer}'")
