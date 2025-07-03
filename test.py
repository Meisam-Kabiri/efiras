import os
from dotenv import load_dotenv
from openai import OpenAI
import langgraph
from langgraph.langgraph import LangGraph


# List all the attributes and methods available in the langgraph module
print(dir(langgraph))

load_dotenv()  # Load environment variables from .env file
openai_api_key = os.getenv("GPT_API_KEY")
print(openai_api_key)
client = OpenAI(api_key=openai_api_key)

def call_gpt(openai_key, prompt: str) -> str:
    client = OpenAI(api_key=openai_key)
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content

call_gpt(openai_api_key, "Hello, how are you?")  # Example call to test the function