import json
from typing import Any, Dict, Optional
from langchain_ollama import ChatOllama
from loguru import logger
from opik.integrations.langchain import OpikTracer
from llm_engineering.application.rag.prompt_templates import SelfQueryTemplate, AIAssistantQueryTemplate

try:
    import boto3
except ModuleNotFoundError:
    logger.warning("Couldn't load AWS or SageMaker imports. Run 'poetry install --with aws' to support AWS.")


from llm_engineering.domain.inference import Inference
from llm_engineering.settings import settings


class LLMInferenceLocal(Inference):
    """
    Class for performing inference using a SageMaker endpoint for LLM schemas.
    """

    def __init__(
        self, model_client
    ) -> None:
        super().__init__()
        self.payload = {
            'parameters':''
        }
        self._model_client = model_client

    def set_payload(self, inputs: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Sets the payload for the inference request.

        Args:
            inputs (str): The input text for the inference.
            parameters (dict, optional): Additional parameters for the inference. Defaults to None.
        """
        self.payload["inputs"] = inputs
        if parameters:
            self.payload["parameters"] = parameters

    def inference(self) -> Dict[str, Any]:
        """
        Performs the inference request using the SageMaker endpoint.

        Returns:
            dict: The response from the inference request.
        Raises:
            Exception: If an error occurs during the inference request.
        """

        prompt = AIAssistantQueryTemplate().create_template()

        chain = prompt | self._model_client

        response = chain.invoke({"user_query_context": self.payload["inputs"]})
        result = response.content
        result = result.split('</think>')[1].strip()
        return result
