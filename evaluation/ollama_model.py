from deepeval.models import DeepEvalBaseLLM
from config.settings import get_settings
from llm.models import LlamaModel


class OllamaDeepEvalModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model_name: str | None = None,
        host: str | None = None,
    ):
        settings = get_settings()
        self.model = LlamaModel(
            model_name=model_name or settings.llm_model,
            host=host or settings.ollama_host,
        )

    def load_model(self):
        return self.model

    def generate(
        self,
        prompt: str,
        schema=None,
    ):
        system_prompt = (
            "You are a strict evaluation judge. Respond ONLY with valid JSON "
            "matching the required schema."
        )
        json_schema = schema.model_json_schema() if schema is not None else None

        response = self.model.generate(
            system_prompt=system_prompt,
            user_prompt=prompt,
            temperature=0.0,
            max_new_tokens=2048,
            format=json_schema,
        )

        if schema is not None:
            try:
                return schema.model_validate_json(response)
            except Exception:
                # Retry once with explicit schema reminder if 3B model malformed JSON
                retry_response = self.model.generate(
                    system_prompt=system_prompt,
                    user_prompt=f"{prompt}\n\nEnsure valid JSON output.",
                    temperature=0.0,
                    max_new_tokens=2048,
                    format=json_schema,
                )
                return schema.model_validate_json(retry_response)

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