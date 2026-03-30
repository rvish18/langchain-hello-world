from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
#print("HF_TOKEN_READ:", os.environ.get('HF_TOKEN_READ'))

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN_READ"],
)

completion = client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct-0905",
    messages=[
        {
            "role": "user",
            "content": "Create a recipe for a chocolate cake."
        }
    ],
)

print(completion.choices[0].message)
print(completion.choices[0].message.content)