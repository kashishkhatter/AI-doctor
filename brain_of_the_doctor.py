# ai-doctor-2.0-voice-and-vision/brain_of_the_doctor.py
import os
import base64
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image_with_query(query, model, encoded_image=None, messages=None):
    client = Groq()

    if messages is None:
        content = [{"type": "text", "text": query}]
        if encoded_image:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                }
            )
        messages = [{"role": "user", "content": content}]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return chat_completion.choices[0].message.content