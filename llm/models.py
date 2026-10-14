from collections.abc import Iterator
from typing import TypeVar
from ollama import Client
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LlamaModel:
    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
    ):
        self.model_name = model_name
        self.client = Client(host=host)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
        format: dict | str | None = None,
    ) -> str:
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": max_new_tokens,
            },
            format=format if format is not None else "",
        )
        return response["message"]["content"].strip()

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float = 0.0,
    ) -> T:
        """
        Forces Ollama to adhere to a Pydantic JSON schema and validates the result.
        """
        raw_json = self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            format=response_model.model_json_schema(),
        )
        return response_model.model_validate_json(raw_json)

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """
        Streams generated tokens one chunk at a time.
        """
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": max_new_tokens,
            },
            stream=True,
        )
        for chunk in response:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token