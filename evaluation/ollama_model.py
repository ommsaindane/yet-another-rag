from deepeval.models import DeepEvalBaseLLM

from llm.models import LlamaModel


class OllamaDeepEvalModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
    ):
        self.model = LlamaModel(
            model_name=model_name,
            host=host,
        )

    def load_model(self):
        return self.model

    def generate(
        self,
        prompt: str,
        schema=None,
    ):

        response = self.model.generate(
            system_prompt=(
                "You are an evaluation model. Respond ONLY with valid JSON."
            ),
            user_prompt=prompt,
            temperature=0.0,
            max_new_tokens=4096,
            format=schema.model_json_schema() if schema else None,
        )

        if schema is not None:
            return schema.model_validate_json(response)

        return response

    async def a_generate(
        self,
        prompt: str,
        schema=None,
    ):
        return self.generate(
            prompt=prompt,
            schema=schema,
        )

    def get_model_name(self) -> str:
        return self.model.model_name

