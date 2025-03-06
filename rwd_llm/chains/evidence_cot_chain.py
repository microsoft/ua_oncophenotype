import re
from typing import Any, Dict, List, Optional, Type, Union

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_openai import ChatOpenAI

from .consistency_chain import LLMConsistencyChain
from .evidence_chain import find_evidence
from .evidence_chain_prompts import (
    EVIDENCE_COT_STOP,
    EvidenceChainExample,
    get_evidence_cot_prompt,
    get_evidence_cot_response_pattern,
)

EVIDENCE_COT_CHAIN_TYPE = "llm_evidence_cot_chain"
EVIDENCE_COT_RESPONSE_PATTERN = get_evidence_cot_response_pattern()


class EvidenceCoTChain(Chain):
    evidence_cot_chain: LLMChain
    label_key: str = "label"

    @classmethod
    def from_openai_llm(
        cls,
        preamble: str,
        question: str,
        llm_class: Type[ChatOpenAI],
        answer_mapping: Union[List[str], Dict[str, Any]],
        llm_extra_args: Optional[Dict[str, Any]] = None,
        examples: List[EvidenceChainExample] = [],
        consistency_temperature: float = 0.0,
        num_consistency_completions: int = 1,
        **kwargs: Any,
    ) -> "EvidenceCoTChain":
        """Create a chain from an OpenAI LLM class"""
        answer_options = (
            list(answer_mapping.keys())
            if isinstance(answer_mapping, dict)
            else answer_mapping
        )
        evidence_prompt = get_evidence_cot_prompt(
            preamble=preamble,
            question=question,
            examples=examples,
            answer_options=answer_options,
        )
        llm_extra_args = llm_extra_args or {}
        evidence_llm = llm_class(
            **llm_extra_args,
            temperature=consistency_temperature,
            verbose=True,
            n=num_consistency_completions,
        )
        evidence_chain = LLMConsistencyChain(
            llm=evidence_llm,
            prompt=evidence_prompt,
            label_mapping=answer_mapping,
            stop=EVIDENCE_COT_STOP,
        )
        return cls(
            evidence_cot_chain=evidence_chain,
            **kwargs,
        )

    @property
    def input_keys(self) -> List[str]:
        """Defines the input keys."""
        # text expected
        return self.evidence_cot_chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        """Defines the output keys."""
        extra_keys = ["evidence", "evidence_string", self.label_key]
        return self.evidence_cot_chain.output_keys + extra_keys

    @property
    def _chain_type(self) -> str:
        return EVIDENCE_COT_CHAIN_TYPE

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        answer_response = self.evidence_cot_chain(
            inputs, return_only_outputs=True, callbacks=callbacks
        )

        response_text = answer_response["text"]

        # parse response
        m = re.match(EVIDENCE_COT_RESPONSE_PATTERN, response_text, re.DOTALL)
        if m is None:
            print(
                "Error: Could not parse"
                f" response.\nPattern:\n{EVIDENCE_COT_RESPONSE_PATTERN}\n"
                f"Text:\n{response_text}\n"
            )
            label = None
            evidence = None
            extracted_string = None
        else:
            evidence_string = m.group(1)
            label = m.group(3)
            evidence = find_evidence(evidence_string, inputs["text"])
            if evidence:
                extracted_string = inputs["text"][evidence.begin : evidence.end]
                if extracted_string != evidence_string:
                    print(
                        f"Warning:\n     '{extracted_string}' !=\n    "
                        f" '{evidence_string}'"
                    )
            else:
                extracted_string = evidence_string

        return {
            **answer_response,
            **{
                self.label_key: label,
                "evidence": evidence,
                "evidence_string": extracted_string,
            },
        }
