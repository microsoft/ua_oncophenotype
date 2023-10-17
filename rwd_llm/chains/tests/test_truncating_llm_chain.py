import os
from typing import Optional

import pytest
import tiktoken
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from llm_lib.dtypes.dtypes import ClinicalNote
from llm_lib.retrieval.retrieval_utils import build_and_get_index
from llm_lib.tests.utils import FakeLLM

from ..truncating_llm_chain import OPENAI_MODEL_TOKEN_MAPPING, TruncatingLLMChain


class FakeOpenAIClient:
    def __init__(self, model_name: str, truncation_message: Optional[str] = None):
        self.model_name = model_name
        self.enc = tiktoken.encoding_for_model(model_name)
        self.truncation_message = truncation_message

    def create(self, prompt: str, max_tokens: int, **kwargs):
        max_context_length = OPENAI_MODEL_TOKEN_MAPPING[self.model_name]
        if isinstance(prompt, list):
            if len(prompt) > 1:
                raise ValueError("Prompt must be a single string")
            prompt = prompt[0]
        if not isinstance(prompt, str):
            raise ValueError(f"Prompt must be a string, got {prompt}")
        # save for testing
        self._prompt = prompt
        llm_prompt = self.enc.encode(prompt)
        n_prompt_tokens = len(llm_prompt)
        ttl_tokens = n_prompt_tokens + max_tokens
        if ttl_tokens > max_context_length:
            raise ValueError(f"Prompt+response is too long: {ttl_tokens} tokens")
        else:
            print(f"Token count okay: {ttl_tokens} tokens")

        fake_response = "A useless response."
        n_fake_response_tokens = len(self.enc.encode(fake_response))

        # fake OpenAI response
        return {
            "usage": {
                "prompt_tokens": n_prompt_tokens,
                "completion_tokens": n_fake_response_tokens,
                "total_tokens": n_prompt_tokens + n_fake_response_tokens,
            },
            "choices": [
                {
                    "text": "A useless response.",
                    "finish_reason": "stop",
                }
            ],
        }


def test_truncating_llm_chain():
    # need to set OPENAI_API_KEY to something, otherwise it will fail
    os.environ["OPENAI_API_KEY"] = "fake"

    template = """
    This is a prompt with a document: {the_doc}.
    """
    model_name = "text-davinci-003"  # max of 4097 tokens

    max_response_tokens = 10

    # create a long document that needs to be truncated
    long_doc = "This is a very long document" * 1000

    prompt = PromptTemplate(input_variables=["the_doc"], template=template)

    llm = OpenAI(
        temperature=0.0,
        n=1,
        model_name=model_name,
        max_tokens=max_response_tokens,
    )
    # replace true client with fake that doesn't actually make API calls
    llm.client = FakeOpenAIClient(model_name=model_name)
    # chain = LLMChain(prompt=prompt, llm=llm)
    chain = TruncatingLLMChain(prompt=prompt, llm=llm)
    inputs = {
        "the_doc": long_doc,
    }
    chain(inputs)
    # just ensure the llm was called
    assert llm.client._prompt


def test_truncating_llm_chain_with_message():
    # need to set OPENAI_API_KEY to something, otherwise it will fail
    os.environ["OPENAI_API_KEY"] = "fake"

    doc_key = "the_doc"
    template = f"""
    This is a prompt with a question: {{the_question}} and a document: {{{doc_key}}}.
    """
    model_name = "text-davinci-003"  # max of 4097 tokens

    max_response_tokens = 10

    # create a long document that needs to be truncated
    long_doc = "This is a very long document" * 1000

    prompt = PromptTemplate(
        input_variables=[doc_key, "the_question"], template=template
    )

    llm = OpenAI(
        temperature=0.0,
        n=1,
        model_name=model_name,
        max_tokens=max_response_tokens,
    )
    # replace true client with fake that doesn't actually make API calls
    llm.client = FakeOpenAIClient(model_name=model_name)
    # chain = LLMChain(prompt=prompt, llm=llm)
    truncation_message = "==TRUNCATED DOCUMENT=="
    chain = TruncatingLLMChain(
        prompt=prompt, llm=llm, doc_key=doc_key, truncation_message=truncation_message
    )
    inputs = {
        doc_key: long_doc,
        "the_question": (
            "This is a somewhat long question, meant to add a bunch of extra tokens."
        ),
    }
    chain(inputs)
    # we saved the prompt in the fake client
    assert llm.client._prompt.find(truncation_message) > -1


def test_truncating_llm_chain_using_apply():
    # need to set OPENAI_API_KEY to something, otherwise it will fail
    os.environ["OPENAI_API_KEY"] = "fake"

    template = """
    This is a prompt with a document: {the_doc}.
    """
    model_name = "text-davinci-003"  # max of 4097 tokens

    max_response_tokens = 10

    # create a long document that needs to be truncated
    long_doc = "This is a very long document" * 1000

    prompt = PromptTemplate(input_variables=["the_doc"], template=template)

    llm = OpenAI(
        temperature=0.0,
        n=1,
        model_name=model_name,
        max_tokens=max_response_tokens,
    )
    # replace true client with fake that doesn't actually make API calls
    llm.client = FakeOpenAIClient(model_name=model_name)
    # chain = LLMChain(prompt=prompt, llm=llm)
    chain = TruncatingLLMChain(prompt=prompt, llm=llm)
    inputs = {
        "the_doc": long_doc,
    }
    chain.apply([inputs])
    # just ensure the llm was called
    assert llm.client._prompt


if __name__ == "__main__":
    test_truncating_llm_chain()
    test_truncating_llm_chain_with_message()
    test_truncating_llm_chain_using_apply()
