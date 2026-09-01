# import openai
import os
import sys

from openai import OpenAI

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from adapt_tracing import wrap_openai_client

from agentdriver.llm_core.api_keys import OPENAI_API_KEY, OPENAI_BASE_URL

# Client dùng chung cho mọi lời gọi LLM của Agent-Driver.
# wrap_openai_client() gắn thêm span "llm" trên Braintrust (no-op nếu tracing tắt).
client = wrap_openai_client(
    OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL or None)
)


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    # print("completion_with_backoff kwargs:", kwargs)
    # input()
    # return openai.ChatCompletion.create(**kwargs)
    return client.chat.completions.create(**kwargs)
