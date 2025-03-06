from typing import Any, Dict, List, Optional, Tuple, Type, Union

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .consistency_chain import LLMConsistencyChain
from .evidence_chain_prompts import (
    ANSWER_STOP,
    EVIDENCE_STOP,
    EvidenceChainExample,
    get_answer_prompt,
    get_evidence_prompt,
)
from .stop_chain import LLMStopChain


def _normalize(input_str: str) -> Tuple[str, List[int]]:
    whitespace = set([" ", "\n", "\t"])
    norm_str = ""
    offset_map = []
    last_is_whitespace = False
    for cidx, c in enumerate(input_str):
        if c in whitespace:
            if last_is_whitespace:
                # skip this character
                continue
            else:
                last_is_whitespace = True
                norm_str += " "
        else:
            last_is_whitespace = False
            norm_str += c
        offset_map.append(cidx)
    # need to add the last offset
    offset_map.append(len(input_str))
    return norm_str, offset_map


class Evidence(BaseModel):
    begin: int
    end: int


EVIDENCE_CHAIN_TYPE = "llm_evidence_chain"


class EvidenceChain(Chain):
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
        label_key: str = "label",
        raw_text_key: str = "text",  # output of answer chain
        examples: List[EvidenceChainExample] = [],
        consistency_temperature: float = 0.7,
        num_consistency_completions: int = 20,
        **kwargs: Any,
    ) -> "EvidenceChain":
        """Create a chain from an OpenAI LLM class"""
        llm_extra_args = llm_extra_args or {}
        answer_options = (
            list(answer_mapping.keys())
            if isinstance(answer_mapping, dict)
            else answer_mapping
        )
        answer_llm = llm_class(
            **llm_extra_args,
            temperature=consistency_temperature,
            verbose=True,
            n=num_consistency_completions,
            best_of=num_consistency_completions,
        )
        answer_prompt = get_answer_prompt(
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
            stop=ANSWER_STOP,
        )
        evidence_prompt = get_evidence_prompt(
            preamble=preamble,
            question=question,
            examples=examples,
            answer_options=answer_options,
        )
        evidence_llm = llm_class(**llm_extra_args, temperature=0.0, verbose=True, n=1)
        # evidence_chain = LLMConsistencyChain(
        #     llm=evidence_llm,
        #     prompt=evidence_prompt,
        #     label_mapping=answer_mapping,
        #     stop=EVIDENCE_STOP,
        # )
        evidence_chain = LLMStopChain(
            llm=evidence_llm,
            prompt=evidence_prompt,
            stop=EVIDENCE_STOP,
        )

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
        return EVIDENCE_CHAIN_TYPE

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        answer_response = self.answer_chain.invoke(
            inputs, return_only_outputs=True, callbacks=callbacks
        )

        answer = answer_response[self.answer_chain.output_key]

        evidence_inputs = {**inputs, **{"answer": answer}}
        callbacks = _run_manager.get_child()
        evidence_response = self.evidence_chain.invoke(
            evidence_inputs, return_only_outputs=True, callbacks=callbacks
        )
        evidence_string = evidence_response[self.evidence_chain.output_key]
        try:
            evidence = find_evidence(evidence_string, inputs["text"])
            if evidence is None:
                print(f"Warning: evidence ({evidence_string}) not found in text")
                extracted_string = evidence_string
            else:
                extracted_string = inputs["text"][evidence.begin : evidence.end]
                if extracted_string != evidence_string:
                    print(
                        f"Warning:\n     '{extracted_string}' !=\n    "
                        f" '{evidence_string}'"
                    )
        except ValueError as e:
            print(f"Error: {e}")
            evidence = None
            extracted_string = evidence_string

        return {
            **answer_response,
            **{"evidence": evidence, "evidence_string": extracted_string},
        }


def find_evidence(evidence_string: str, doc_text: str) -> Optional[Evidence]:
    """Find the evidence string in the document text.  Raises ValueError if not
    found."""
    begin = doc_text.find(evidence_string)
    if begin > -1:
        end = begin + len(evidence_string)
    else:
        # try again, normalizing whitespace
        norm_text, offset_map = _normalize(doc_text)
        norm_str, _ = _normalize(evidence_string)
        norm_begin = norm_text.lower().find(norm_str.lower())
        if norm_begin == -1:
            # TODO: maybe one more try with edit distance
            return None
        begin = offset_map[norm_begin]
        end = offset_map[norm_begin + len(norm_str)]
    return Evidence(begin=begin, end=end)
