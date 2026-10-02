"""LLM access through LiteLLM (any provider LiteLLM supports, e.g. ``mistral/codestral-2508``)."""
import logging
import time

import tiktoken
from litellm import completion

MAX_ATTEMPTS = 100


def chat_completion(messages, model, temperature=1.0, n=1):
    """Query ``model`` with retries; returns the list of choices."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return completion(model=model, messages=messages, temperature=temperature, n=n).choices
        except Exception as e:
            logging.info(f"LLM attempt {attempt + 1} failed with error: {e}")
            time.sleep(3)
    raise RuntimeError(f"LLM call failed {MAX_ATTEMPTS} times in a row, giving up.")


class LLMClient:
    """Single-turn chat client that also tracks token usage."""

    def __init__(self, model, temperature=1.0):
        self.model = model
        self.temperature = temperature
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def get_response(self, prompt_content):
        messages = [{"role": "user", "content": prompt_content}]
        response = chat_completion(messages, self.model, self.temperature)[0].message.content
        self.prompt_tokens += len(self._encoding.encode(prompt_content))
        self.completion_tokens += len(self._encoding.encode(response))
        logging.info(f"LLM usage: prompt_tokens = {self.prompt_tokens}, completion_tokens = {self.completion_tokens}")
        return response
