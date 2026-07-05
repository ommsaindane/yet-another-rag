from pathlib import Path

import yaml


class PromptManager:
    def __init__(
        self,
        prompt_file: str | Path = "config/prompts.yaml",
    ):
        with open(prompt_file, "r", encoding="utf-8") as f:
            self.prompts = yaml.safe_load(f)

    def get(
        self,
        name: str,
        **kwargs,
    ) -> tuple[str, str]:

        if name not in self.prompts:
            raise ValueError(f"Unknown prompt '{name}'.")

        prompt = self.prompts[name]

        system_prompt = prompt["system"]

        user_prompt = prompt["user"].format(**kwargs)

        return system_prompt, user_prompt
