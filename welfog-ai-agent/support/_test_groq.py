import os
import time

import requests
from dotenv import load_dotenv

load_dotenv(r"c:\Users\arjun\welfog\welfog-ai\welfog-ai-agent\.env")
key = os.getenv("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/chat/completions"
payload = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {
            "role": "system",
            "content": 'Return JSON only: {"intent":"order_history","account_list_kind":"purchase_history_in_chat"}',
        },
        {"role": "user", "content": "bhai mereko meri order history dikha"},
    ],
    "response_format": {"type": "json_object"},
    "max_tokens": 200,
    "temperature": 0,
}
t0 = time.time()
r = requests.post(
    url,
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    json=payload,
    timeout=15,
)
print("status", r.status_code, "elapsed", round(time.time() - t0, 2))
print(r.text[:600])
