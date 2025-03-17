import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import tiktoken
from langchain.chains.llm import LLMChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import BasePromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_openai.llms.base import BaseOpenAI
from langchain_text_splitters import TokenTextSplitter
from pydantic import model_validator

logger = logging.getLogger(__name__)

# just copied from langchain
OPENAI_MODEL_TOKEN_MAPPING = {
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-35-turbo": 4096,
    "gpt-35-turbo-16k": 16384,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "text-ada-001": 2049,
    "ada": 2049,
    "text-babbage-001": 2040,
    "babbage": 2049,
    "text-curie-001": 2049,
    "curie": 2049,
    "davinci": 2049,
    "text-davinci-003": 4097,
    "text-davinci-002": 4097,
    "code-davinci-002": 8001,
    "code-davinci-001": 8001,
    "code-cushman-002": 2048,
    "code-cushman-001": 2048,
}


def _get_max_context(model_name) -> int:
    return OPENAI_MODEL_TOKEN_MAPPING[model_name]


DEFAULT_MAX_COMPLETION_TOKENS = 256


class OpenAIDocumentTruncator:
    def __init__(
        self,
        model_name: str,
        template: BasePromptTemplate,
        truncation_message: Optional[str] = None,
        doc_key: Optional[str] = None,
        buffer: int = 1,  # extra tokens to leave for safety
    ):
        self.model_name = model_name
        self.truncation_message = truncation_message
        self.buffer = buffer
        self.encoder = tiktoken.encoding_for_model(model_name)
        empty_inputs = {k: "" for k in template.input_variables}
        empty_prompt = template.format(**empty_inputs)
        self.n_test_prompt_tokens = len(self.encoder.encode(empty_prompt))
        self.max_context = _get_max_context(model_name)
        self.n_truncation_message_tokens = (
            len(self.encoder.encode(truncation_message)) if truncation_message else 0
        )
        if doc_key is None:
            if len(template.input_variables) == 1:
                doc_key = template.input_variables[0]
            else:
                raise ValueError(
                    "doc_key must be specified if there are multiple input variables:"
                    f" {template.input_variables}"
                )
        self.doc_key = doc_key
        self.other_inputs = [k for k in template.input_variables if k != doc_key]

    def get_max_doc_tokens(self, max_completion_tokens: int) -> int:
        used_tokens = self.n_test_prompt_tokens + self.buffer + max_completion_tokens
        return self.max_context - used_tokens

    def truncate(self, inputs: dict, max_completion_tokens: int) -> dict:
        doc = inputs[self.doc_key]
        n_other_input_tokens = sum(
            [len(self.encoder.encode(inputs[k])) for k in self.other_inputs]
        )
        max_doc_tokens = self.get_max_doc_tokens(max_completion_tokens)
        max_doc_tokens -= n_other_input_tokens
        if max_doc_tokens < self.n_truncation_message_tokens:
            # nothing we can do if the prompt alone is too long
            raise ValueError(
                f"Prompt tokens are longer than max context ({self.max_context}) even"
                " without the document"
            )
        n_doc_tokens = len(self.encoder.encode(doc))
        if n_doc_tokens > max_doc_tokens:
            logger.warning(
                f"Truncating document from {n_doc_tokens} to {max_doc_tokens}"
            )
            splitter = TokenTextSplitter(
                model_name=self.model_name,
                chunk_size=max_doc_tokens - self.n_truncation_message_tokens,
            )
            # truncate by taking only the first chunk
            doc = splitter.split_text(doc)[0] + f" {self.truncation_message}"
            inputs[self.doc_key] = doc
        return inputs


class TruncatingLLMChain(LLMChain):
    truncate: bool = True
    doc_key: Optional[str] = None
    truncation_message: Optional[str] = None
    truncator: Optional[OpenAIDocumentTruncator] = None

    @classmethod
    def _set_doc_key(cls, doc_key: Optional[str], prompt: BasePromptTemplate) -> str:
        if doc_key is None:
            if len(prompt.input_variables) == 1:
                doc_key = prompt.input_variables[0]
            else:
                raise ValueError(
                    "doc_key not specified and there are"
                    f" multiple input variables ({list(prompt.input_variables)})"
                )
        return doc_key

    @model_validator(mode="before")
    def set_truncator(cls, data: Dict[str, Any]) -> dict:
        truncator: Optional[OpenAIDocumentTruncator] = data.get("truncator")
        llm: Union[BaseOpenAI, BaseChatOpenAI] = data["llm"]
        template: BasePromptTemplate = data["prompt"]
        truncation_message: Optional[str] = data.get("truncation_message")
        doc_key: str = cls._set_doc_key(data.get("doc_key"), template)

        if truncator is None:
            truncator = OpenAIDocumentTruncator(
                model_name=llm.model_name,
                template=template,
                truncation_message=truncation_message,
                doc_key=doc_key,
            )
        data["truncator"] = truncator
        return data

    def get_doc_key(self) -> str:
        if self.doc_key is None:
            raise ValueError("doc_key not set (should have been set in root_validator)")
        return self.doc_key

    def _truncate(self, inputs: dict, max_completion_tokens: int) -> dict:
        if self.truncator is None:
            raise ValueError(
                "truncator not set (should have been set in root_validator)"
            )
        return self.truncator.truncate(inputs, max_completion_tokens)

    def _get_llm(self) -> Union[BaseOpenAI, BaseChatOpenAI]:
        if isinstance(self.llm, (BaseOpenAI, BaseChatOpenAI)):
            return self.llm
        raise ValueError(
            f"llm must be an OpenAI or ChatOpenAI model, got {type(self.llm)}"
        )

    def _get_model_name(self) -> str:
        return self._get_llm().model_name

    def _get_completion_max_tokens(self) -> int:
        return self._get_llm().max_tokens or DEFAULT_MAX_COMPLETION_TOKENS

    def _get_max_context(self) -> int:
        return OPENAI_MODEL_TOKEN_MAPPING[self._get_model_name()]

    def _truncate_doc(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self.truncate:
            inputs = self._truncate(inputs, self._get_completion_max_tokens())
        return inputs

    # prep_inputs doesn't always get called (e.g. if we use the 'apply' function), so we
    # put the truncation logic here.
    def prep_prompts(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        input_list = [self._truncate_doc(inputs) for inputs in input_list]
        return super().prep_prompts(input_list, run_manager=run_manager)
