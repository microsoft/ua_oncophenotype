from typing import Any, Dict, List, Optional, Type, Union

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.prompts import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.chat import BaseMessagePromptTemplate
from langchain_openai import ChatOpenAI

from .consistency_chain import LLMConsistencyChain
from .evidence_chain import find_evidence
from .evidence_chain_prompts import EvidenceChainExample, _get_answer_prefix_fmt

CHAT_EVIDENCE_CHAIN_TYPE = "llm_chat_evidence_chain"
evidence_prompt = "Evidence:"


class ChatEvidenceChain(Chain):
    answer_chain: LLMChain
    evidence_chain: LLMChain

    @classmethod
    def from_openai_llm(
        cls,
        preamble: str,
        question: str,
        llm_class: Type[ChatOpenAI],
        answer_mapping: Union[List[str], Dict[str, Any]],
        llm_extra_args: Optional[Dict[str, Any]] = None,
        raw_text_key: str = "text",  # output of answer chain
        label_key: str = "label",
        examples: List[EvidenceChainExample] = [],
        temperature: float = 0.0,
        n: int = 1,
        **kwargs: Any,
    ) -> "ChatEvidenceChain":
        """Create a chain from an OpenAI LLM class"""
        llm_extra_args = llm_extra_args or {}
        answer_options = (
            list(answer_mapping.keys())
            if isinstance(answer_mapping, dict)
            else answer_mapping
        )
        answer_llm = llm_class(
            **llm_extra_args,
            temperature=temperature,
            verbose=True,
            n=n,
        )
        answer_prompt = get_chat_prompt(
            preamble=preamble,
            question=question,
            examples=examples,
            answer_options=answer_options,
        )
        answer_chain = LLMConsistencyChain(
            llm=answer_llm,
            prompt=answer_prompt,
            output_key=raw_text_key,
            label_key=label_key,
            label_mapping=answer_mapping,
            stop=["\n"],
        )
        # answer_chain = LLMChain(llm=answer_llm, prompt=answer_prompt)
        evidence_prompt = get_chat_prompt(
            preamble=preamble,
            question=question,
            examples=examples,
            answer_options=answer_options,
            with_answer=True,
        )
        evidence_llm = llm_class(**llm_extra_args, temperature=0.0, verbose=True, n=1)
        evidence_chain = LLMChain(llm=evidence_llm, prompt=evidence_prompt)

        return cls(
            answer_chain=answer_chain,
            evidence_chain=evidence_chain,
            **kwargs,
        )

    @property
    def input_keys(self) -> List[str]:
        """Defines the input keys."""
        # text expected
        return self.answer_chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Defines the output keys."""
        return self.answer_chain.output_keys + ["evidence", "evidence_string"]

    @property
    def _chain_type(self) -> str:
        return CHAT_EVIDENCE_CHAIN_TYPE

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        answer_response = self.answer_chain(
            inputs, return_only_outputs=True, callbacks=callbacks
        )

        answer = answer_response[self.answer_chain.output_key]

        # self.callback_manager.on_text(
        #     text="answer: " + answer + "\n\n",
        #     verbose=self.verbose,
        #     color="yellow",
        # )

        evidence_inputs = {**inputs, **{"answer": answer}}
        callbacks = _run_manager.get_child()
        evidence_response = self.evidence_chain(
            evidence_inputs, return_only_outputs=True, callbacks=callbacks
        )
        evidence_string = evidence_response["text"]
        evidence = find_evidence(evidence_string, inputs["text"])
        if evidence is None:
            print(f"Error: evidence not found: {evidence_string}")
            evidence = None
            extracted_string = evidence_string
        else:
            extracted_string = inputs["text"][evidence.begin : evidence.end]
            if extracted_string != evidence_string:
                print(
                    f"Warning:\n     '{extracted_string}' !=\n     '{evidence_string}'"
                )

        return {
            **answer_response,
            **{"evidence": evidence, "evidence_string": extracted_string},
        }


def get_chat_prompt(
    preamble: str,
    question: str,
    examples: List[EvidenceChainExample],
    answer_options: List[str],
    with_answer: bool = False,
) -> ChatPromptTemplate:
    messages: List[BaseMessagePromptTemplate] = []
    messages.append(SystemMessagePromptTemplate.from_template(preamble))
    question_fmt = _get_answer_prefix_fmt(
        question=question, answer_options=answer_options
    )
    for example in examples:
        ex_doc = example.text
        ex_answer = example.answer
        ex_evidence = example.evidence
        ex_question = question_fmt.format(text=ex_doc)
        messages.append(
            SystemMessagePromptTemplate.from_template(
                ex_question, additional_kwargs={"name": "example_user"}
            )
        )
        messages.append(
            SystemMessagePromptTemplate.from_template(
                ex_answer, additional_kwargs={"name": "example_assistant"}
            )
        )
        messages.append(
            SystemMessagePromptTemplate.from_template(
                evidence_prompt, additional_kwargs={"name": "example_user"}
            )
        )
        messages.append(
            SystemMessagePromptTemplate.from_template(
                ex_evidence, additional_kwargs={"name": "example_assistant"}
            )
        )

    messages.append(HumanMessagePromptTemplate.from_template(question_fmt))
    if with_answer:
        messages.append(AIMessagePromptTemplate.from_template("{answer}"))
        messages.append(HumanMessagePromptTemplate.from_template(evidence_prompt))
    chat_prompt = ChatPromptTemplate.from_messages(messages)
    return chat_prompt
