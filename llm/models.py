from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)


class LlamaModel:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )

        self.pipeline = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]

        outputs = self.pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        generated = outputs[0]["generated_text"]

        if isinstance(generated, list):
            return generated[-1]["content"].strip()

        return generated.strip()

