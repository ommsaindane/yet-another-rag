from ollama import Client


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
        format: dict | None = None,
    ) -> str:

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            options={
                "temperature": temperature,
                "num_predict": max_new_tokens,
            },
            format=format if format is not None else "",
        )

        return response["message"]["content"].strip()
