import openai
from dotenv import load_dotenv
import os

load_dotenv()
model = os.getenv("OPENAI_MODEL")
base_url = os.getenv("OPENAI_ENDPOINT")
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key, base_url=base_url)

# GPTのテスト
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "こんにちは！"}],
)
print(response.choices[0].message.content)
