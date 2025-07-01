import opik
from langchain_ollama import ChatOllama
from loguru import logger
import json
from llm_engineering.domain.queries import Query
from llm_engineering.settings import settings
from .base import RAGStep
from .prompt_templates import QueryExpansionTemplate


class QueryExpansion(RAGStep):
    @opik.track(name="QueryExpansion.generate")
    def generate(self, model, query: Query, expand_to_n: int) -> list[Query]:
        assert expand_to_n > 0, f"'expand_to_n' should be greater than 0. Got {expand_to_n}."

        if self._mock:
            return [query for _ in range(expand_to_n)]

        query_expansion_template = QueryExpansionTemplate()
        prompt = query_expansion_template.create_template(expand_to_n - 1)

        #model = ChatOllama(model=settings.OLLAMA_MODEL_ID)

        chain = prompt | model

        response = chain.invoke({"question": query})
        result = response.content
        result = result.split('</think>')[1].strip().replace('```json', '').replace('```', '')
        query_metadata = json.loads(str(result))

        queries = [query]
        queries += [
            query.replace_content(stripped_content)
            for content in query_metadata['alternatives']
            if (stripped_content := content.strip())
        ]

        return queries


if __name__ == "__main__":
    query = Query.from_str("Write an article about the best types of advanced RAG methods.")
    query_expander = QueryExpansion()
    expanded_queries = query_expander.generate(query, expand_to_n=3)
    for expanded_query in expanded_queries:
        logger.info(expanded_query.content)
