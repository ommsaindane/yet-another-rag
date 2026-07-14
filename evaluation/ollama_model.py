from typing import Optional

from deepeval.models import DeepEvalBaseLLM

from llm.models import LlamaModel


class OllamaDeepEvalModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.host = host
        self._model: Optional[LlamaModel] = None

    def load_model(self) -> LlamaModel:
        if self._model is None:
            self._model = LlamaModel(
                model_name=self.model_name,
                host=self.host,
            )
        return self._model

    def generate(
        self,
        prompt: str,
        schema=None,
    ):
        model = self.load_model()

        response = model.generate(
            system_prompt="You are a helpful evaluation assistant.",
            user_prompt=prompt,
            temperature=0.0,
        )

        if schema is not None:
            return schema.model_validate_json(response)

        return response

    async def a_generate(
        self,
        prompt: str,
        schema=None,
    ):
        # Ollama client is synchronous, so reuse generate().
        return self.generate(
            prompt=prompt,
            schema=schema,
        )

    def get_model_name(self) -> str:
        return self.model_name
