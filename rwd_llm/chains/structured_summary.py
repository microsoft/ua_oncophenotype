import logging
from typing import Any, Dict, List, Optional

from langchain.callbacks.base import Callbacks
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.documents import Document
from langchain_core.memory import BaseMemory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from rwd_llm.output_parsers.pydantic_parser_with_examples import (
    PydanticOutputParserWithExamples,
)

from ..utils.chain_utils import get_child_config
from .evidence_chain import find_evidence
from .question_with_structured_output import chat_prompt_with_structured_output
from .truncating_llm_chain import TruncatingLLMChain

logger = logging.getLogger(__name__)


#
# Formatting used by GPT
#
class SummaryEvidence(BaseModel):
    evidence: str = Field(description="evidence text")


class SummaryFinding(BaseModel):
    evidence: List[SummaryEvidence] = Field(
        description="Evidence span(s) supporting the finding"
    )
    finding: str = Field(description="Extracted fact from the source document")


class Summary(BaseModel):
    title: str = Field(
        description="one-line 'title' summarizing the purpose of the source document"
    )
    findings: List[SummaryFinding] = Field(
        description="Extracted findings from the source document"
    )


#
# Parsed formatting with [begin, end) spans
#
class ParsedSummaryEvidence(SummaryEvidence):
    doc_id: str
    begin: Optional[int] = None
    end: Optional[int] = None


DEFAULT_FINDING_FORMAT = "  - {finding}"
DEFAULT_SUMMARY_FORMAT = """
{title}
{findings}
"""


def validate_evidence(summary: Summary, doc: Document) -> Summary:
    summary = summary.model_copy()
    # verify the summary evidence
    for finding in summary.findings:
        parsed_evidence = []
        for evidence in finding.evidence:
            parsed_span = ParsedSummaryEvidence(
                evidence=evidence.evidence, doc_id=doc.metadata["id"]
            )
            # get the indices of the evidence span
            ev_span = find_evidence(evidence.evidence, doc.page_content)
            parsed_evidence.append(parsed_span)
            if ev_span is None:
                logger.warning(f"evidence {evidence.evidence} not found in doc")
            else:
                parsed_span.begin = ev_span.begin
                parsed_span.end = ev_span.end
        finding.evidence = parsed_evidence
    return summary


DEFAULT_EXAMPLE = Summary(
    title="Title summarizing the purpose of the source document",
    findings=[
        SummaryFinding(
            evidence=[
                SummaryEvidence(
                    evidence="Exact span from input supporting this finding.",
                ),
                SummaryEvidence(
                    evidence="Another exact span from input supporting this finding.",
                ),
            ],
            finding="Extracted fact from the source document",
        ),
        SummaryFinding(
            evidence=[
                SummaryEvidence(
                    evidence="Exact span from input supporting this finding.",
                ),
            ],
            finding="Another extracted fact from the source document",
        ),
    ],
)

DEFAULT_PARSER = PydanticOutputParserWithExamples(
    pydantic_object=Summary, examples=DEFAULT_EXAMPLE
)


class StructuredDocumentSummaryChain(Chain):
    llm_chain: Optional[LLMChain] = None
    doc_key: str = "doc"
    prompt_doc_key: str = "text"
    parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=Summary)
    findings_format: str = DEFAULT_FINDING_FORMAT
    summary_format: str = DEFAULT_SUMMARY_FORMAT
    output_key: str = "summary"
    concurrent_calls: bool = True

    @property
    def summary_chain(self) -> LLMChain:
        if self.llm_chain is None:
            raise ValueError("llm_chain is not set")
        return self.llm_chain

    @property
    def input_keys(self) -> List[str]:
        summary_chain_keys = list(self.summary_chain.input_keys)
        if self.prompt_doc_key not in summary_chain_keys:
            raise ValueError(
                f"expected prompt_doc_key {self.prompt_doc_key} in"
                " summary_chain.input_keys"
            )
        summary_chain_keys.remove(self.prompt_doc_key)
        summary_chain_keys.append(self.doc_key)
        return summary_chain_keys

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    @classmethod
    def from_openai_llm(
        cls,
        llm: ChatOpenAI,
        question: str,
        input_variables: List[str],
        preamble: Optional[str] = None,
        instructions: Optional[str] = None,
        format_instructions_variable: str = "format_instructions",
        memory: Optional[BaseMemory] = None,
        doc_key: str = "doc",
        prompt_doc_key: str = "text",
        output_key="summary",
        truncate: bool = True,
        few_shot_examples: Optional[List[dict]] = None,
        example_output_var: str = "result",
        example_indent: int = 2,
        concurrent_calls: bool = True,
    ) -> "StructuredDocumentSummaryChain":
        parser: PydanticOutputParser = DEFAULT_PARSER
        few_shot_examples = few_shot_examples or []  # ensure it's not None
        prompt = chat_prompt_with_structured_output(
            parser=parser,
            preamble=preamble,
            question=question,
            instructions=instructions,
            input_variables=input_variables,
            format_instructions_variable=format_instructions_variable,
            examples=few_shot_examples,
            example_output_var=example_output_var,
            example_indent=example_indent,
        )
        llm_chain = TruncatingLLMChain(
            prompt=prompt, llm=llm, memory=memory, truncate=truncate
        )
        return cls(
            llm_chain=llm_chain,
            memory=memory,
            doc_key=doc_key,
            prompt_doc_key=prompt_doc_key,
            output_key=output_key,
            concurrent_calls=concurrent_calls,
        )

    def _replace_doc(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # replace doc input with text
        doc: Document = inputs[self.doc_key]
        llm_inputs = inputs.copy()
        llm_inputs[self.prompt_doc_key] = doc.page_content
        return llm_inputs

    def _create_response(
        self, doc: Document, response: Dict[str, Any]
    ) -> Dict[str, Any]:
        # copy, don't modify the original doc
        doc = doc.model_copy()
        raw_response = response[self.summary_chain.output_key]
        summary: Summary = self.parser.parse(raw_response)
        summary = validate_evidence(summary, doc)
        # place formatted summary in doc, and store the structured summary in the
        # metadata
        findings_summary = "\n".join(
            self.findings_format.format(finding=finding.finding)
            for finding in summary.findings
        )
        summary_text = self.summary_format.format(
            title=summary.title, findings=findings_summary
        )
        doc.page_content = summary_text
        doc.metadata["summary"] = summary
        # return the doc in output_key
        output = {self.output_key: doc}
        return output

    def apply(
        self, input_list: List[Dict[str, Any]], callbacks: Callbacks = None
    ) -> List[Dict[str, str]]:
        """summarization is often called in parallel, let's push 'apply' down to the
        summary_chain"""
        llm_inputs = [self._replace_doc(inputs) for inputs in input_list]
        if self.concurrent_calls:
            responses = self.summary_chain.apply(llm_inputs, callbacks=callbacks)
        else:
            responses = [
                self.summary_chain(inputs, callbacks=callbacks) for inputs in llm_inputs
            ]
        outputs = [
            self._create_response(input_list[response_idx][self.doc_key], response)
            for response_idx, response in enumerate(responses)
        ]
        return outputs

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        llm_input = self._replace_doc(inputs)

        # generate structured summary
        config = get_child_config(run_manager)
        response = self.summary_chain.invoke(llm_input, config=config)
        output = self._create_response(inputs[self.doc_key], response)
        return output
