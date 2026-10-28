from langchain_ollama import ChatOllama
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

model = ChatOllama(model="llama3.1:latest", temperature=0.1 ,num_gpu=24)

class Review(TypedDict):
    summary: str
    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    "The hardware is great, but the software feels bloated. There are too many "
    "pre-installed apps that I can't remove. Also, the UI looks outdated compared "
    "to other brands. Hoping for a software update to fix this."
)

print(result)
print(result["summary"])
print(result["sentiment"])