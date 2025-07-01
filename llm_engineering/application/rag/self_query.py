import opik
from langchain_ollama import ChatOllama
from loguru import logger
import json
from llm_engineering.domain.queries import Query
from llm_engineering.settings import settings

from .base import RAGStep
from .prompt_templates import SelfQueryTemplate

class SelfQuery(RAGStep):
    @opik.track(name="SelfQuery.generate")
    def generate(self, model, query: Query) -> Query:
        if self._mock:
            return query

        prompt = SelfQueryTemplate().create_template()

        # model = ChatOllama(model=settings.OLLAMA_MODEL_ID)

        chain = prompt | model

        response = chain.invoke({"question": query})

        logger.info(f"Response: {response}")
        result = response.content

        if result == "none":
            return query

        result = result.split('</think>')[1].strip().replace('```json', '').replace('```', '')
        # json format and keys
        query_metadata = json.loads(str(result))
        # get query metadata
        query.author = query_metadata['authors']
        query.topics = query_metadata['topics']
        query.tools = query_metadata['tools']

        return query


if __name__ == "__main__":
    query = Query.from_str("What is generative model?")
    self_query = SelfQuery()
    query = self_query.generate(query)
    logger.info(f"Extracted author: {query.author}")
    logger.info(f"Extracted topics: {query.topics}")
    logger.info(f"Extracted tools: {query.tools}")
