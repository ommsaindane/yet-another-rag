from llm.models import LlamaModel

llm = LlamaModel()

response = llm.generate(
    system_prompt="You are a helpful assistant.",
    user_prompt="Say hello.",
)

print(response)
