"""Hosted OpenAI-compatible model adapter implementing the local LlamaModel API."""
import json
import os
from collections.abc import Iterator
from typing import TypeVar
from openai import OpenAI
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class HostedModel:
    def __init__(self, model_name=None, host=None):
        self.model_name = os.environ.get('HOSTED_LLM_MODEL', 'meta-llama/Llama-3.1-8B-Instruct:fastest')
        token = os.environ.get('HF_TOKEN')
        if not token:
            raise RuntimeError('Set HF_TOKEN in the Space secrets')
        self.client = OpenAI(base_url='https://router.huggingface.co/v1', api_key=token)

    def generate(self, system_prompt, user_prompt, max_new_tokens=1024, temperature=0.2, format=None):
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
        if format is not None:
            messages[0]['content'] += '\nRespond with ONLY valid JSON matching this schema: ' + json.dumps(format)
        response = self.client.chat.completions.create(model=self.model_name, messages=messages, max_tokens=max_new_tokens, temperature=temperature)
        return (response.choices[0].message.content or '').strip()

    def generate_structured(self, system_prompt, user_prompt, response_model: type[T], temperature=0.0) -> T:
        raw = self.generate(system_prompt, user_prompt, temperature=temperature, format=response_model.model_json_schema())
        if raw.startswith('```'):
            raw = raw.strip('`').removeprefix('json').strip()
        return response_model.model_validate_json(raw)

    def stream(self, system_prompt, user_prompt, max_new_tokens=1024, temperature=0.2) -> Iterator[str]:
        response = self.client.chat.completions.create(model=self.model_name, messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}], max_tokens=max_new_tokens, temperature=temperature, stream=True)
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token:
                yield token
