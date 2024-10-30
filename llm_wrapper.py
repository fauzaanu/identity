from typing import TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def send_llm_request(
    model: str,
    system_prompt: str,
    prompt: str,
    response_model: type[T],
    images=list[str],
) -> T:
    """
    Send a prompt to the LLM and return the response
    """
    load_dotenv()
    client = OpenAI()
    user_content = [{"type": "text", "text": prompt}]
    if images:
        user_content.extend(
            {"type": "image_url", "image_url": {"url": url}} for url in images
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    completion = (
        client.beta.chat.completions.parse(
            model=model, messages=messages, response_format=response_model
        )
        .choices[0]
        .message
    )

    if completion.parsed:
        return completion.parsed

    if completion.refusal:
        raise Exception(completion.refusal)
