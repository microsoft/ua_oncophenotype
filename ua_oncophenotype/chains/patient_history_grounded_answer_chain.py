import logging
from typing import Any, Dict, List, Optional, Union, cast

from langchain.chains.llm import LLMChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.documents import Document
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.memory import BaseMemory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, model_validator
from ua_oncophenotype.output_parsers import PydanticOutputParserWithExamples

from .categorical_chain import normalize_label_mapping, parse_output
from .evidence_chain import find_evidence
from .question_with_structured_output import chat_prompt_with_structured_output

logger = logging.getLogger(__name__)


class PatientHistoryEvidence(BaseModel):
    """Raw evidence returned by the LLM (raw string as evidence)"""

    note_id: str = Field(description="note id")
    evidence: str = Field(description="evidence text")


class ParsedPatientHistoryEvidence(PatientHistoryEvidence):
    """Parsed verison of evidence with [begin, end) span"""

    begin: Optional[int] = None
    end: Optional[int] = None


class PatientHistoryGroundedAnswer(BaseModel):
    evidence: List[PatientHistoryEvidence] = Field(
        description="Evidence spans from the patient history supporting the answer"
    )
    reasoning: str = Field(description="Reasoning leading to the answer")
    contradictory_evidence: List[PatientHistoryEvidence] = Field(
        description=(
            "Optional evidence spans from the patient history potentially contradicting"
            " the answer"
        )
    )
    answer: str = Field(description="The answer to the question")


DEFAULT_EXAMPLE = PatientHistoryGroundedAnswer(
    evidence=[
        PatientHistoryEvidence(
            note_id="1",
            evidence="Exact span from input supporting the answer.",
        ),
        PatientHistoryEvidence(
            note_id="3",
            evidence="Another exact span from input supporting the answer.",
        ),
    ],
    reasoning="The step-by-step reasoning leading to the answer.",
    contradictory_evidence=[
        PatientHistoryEvidence(
            note_id="2",
            evidence="Exact span from input that may contradict the answer.",
        ),
    ],
    answer="The answer to the question.",
)

DEFAULT_PARSER = PydanticOutputParserWithExamples(
    pydantic_object=PatientHistoryGroundedAnswer, examples=DEFAULT_EXAMPLE
)


def validate_evidence_from_summarized_notes(
    answer: PatientHistoryGroundedAnswer, summarized_notes: List[Document]
) -> PatientHistoryGroundedAnswer:
    answer = answer.model_copy()
    idx_to_note: Dict[str, Document] = {
        str(note_idx): note for note_idx, note in enumerate(summarized_notes)
    }
    parsed_evidence = []
    for evidence in answer.evidence:
        note_id = str(evidence.note_id)
        if note_id not in idx_to_note:
            logger.warning(f"Note index {note_id} not found in summarized notes")
            continue
        note: Document = idx_to_note[evidence.note_id]
        note_text = note.page_content
        ev = find_evidence(evidence.evidence, note_text)
        parsed_span = ParsedPatientHistoryEvidence(
            evidence=evidence.evidence, note_id=note.metadata["id"]
        )
        if ev is None:
            logger.warning(
                f"Evidence {evidence.evidence} not found in summarized note."
            )
        else:
            parsed_span.begin = ev.begin
            parsed_span.end = ev.end
        parsed_evidence.append(parsed_span)
    answer.evidence = parsed_evidence
    return answer


def validate_evidence_from_summary_text(
    answer: PatientHistoryGroundedAnswer, summary_text: str
) -> PatientHistoryGroundedAnswer:
    answer = answer.model_copy()
    parsed_evidence = []
    for evidence in answer.evidence:
        ev = find_evidence(evidence.evidence, summary_text)
        parsed_span = ParsedPatientHistoryEvidence(
            evidence=evidence.evidence, note_id=evidence.note_id
        )
        if ev is None:
            logger.warning(
                f"Evidence {evidence.evidence} not found in summarized history"
            )
        else:
            parsed_span.begin = ev.begin
            parsed_span.end = ev.end
        parsed_evidence.append(parsed_span)
    answer.evidence = parsed_evidence
    return answer


class PatientHistoryGroundedAnswerChain(LLMChain):
    summarized_notes_key: Optional[str] = None
    """If available, use for grounding answers via evidence"""
    patient_history_key: Optional[str] = None
    """prompt input variable to inject the summarized patient history"""
    answer_key: str = "answer"
    parsed_answer_key: str = "parsed_answer"
    label_mapping: Optional[Union[List[str], Dict[str, str]]] = None
    parser: PydanticOutputParser = DEFAULT_PARSER

    @model_validator(mode="before")
    def _get_patient_history_key(cls, data) -> dict:
        patient_history_key = data.get("patient_history_key")
        # if there is only one input to the prompt, use it as the patient history key
        if patient_history_key is None:
            if len(data["prompt"].input_variables) == 1:
                patient_history_key = data["prompt"].input_variables[0]
            else:
                raise ValueError(
                    "patient_history_key must be specified if there are multiple inputs"
                    " to the prompt"
                )
        else:
            if patient_history_key not in data["prompt"].input_variables:
                raise ValueError(
                    f"patient_history_key {patient_history_key} not found in prompt"
                    " input_variables"
                )
        data["patient_history_key"] = patient_history_key
        return data

    @field_validator("label_mapping")
    @classmethod
    def _normalize_label_mapping(cls, label_mapping):
        # make sure label mapping is a dict with lowercase keys
        if label_mapping:
            return normalize_label_mapping(label_mapping)
        return None

    @property
    def input_keys(self) -> List[str]:
        input_keys = list(self.prompt.input_variables)
        if self.summarized_notes_key is not None:
            input_keys.append(self.summarized_notes_key)
        return input_keys

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key, self.answer_key, self.parsed_answer_key]

    @property
    def normalized_label_mapping(self) -> Optional[Dict[str, str]]:
        if not self.label_mapping:
            return None
        if not isinstance(self.label_mapping, dict):
            raise ValueError(
                "label_mapping should have been normalized by the validator (got list,"
                " expected dict)"
            )
        return self.label_mapping

    @classmethod
    def from_openai_llm(
        cls,
        llm: ChatOpenAI,
        question: str,
        input_variables: List[str],
        patient_history_key: Optional[str] = None,
        summarized_notes_key: Optional[str] = None,
        instructions: Optional[str] = None,
        preamble: Optional[str] = None,
        format_instructions_variable: str = "format_instructions",
        memory: Optional[BaseMemory] = None,
        output_key="answer_raw",
        answer_key="answer",
        parsed_answer_key="parsed_answer",
        label_mapping: Optional[Union[List[str], Dict[str, str]]] = None,
        examples: Optional[Union[List[dict], BaseExampleSelector]] = None,
        example_input_vars: Optional[List[str]] = None,
        example_output_var: str = "result",
        example_indent: int = 2,
        jinja: bool = False,
        use_system_prompt: bool = True,
    ) -> "PatientHistoryGroundedAnswerChain":
        parser: PydanticOutputParser = DEFAULT_PARSER
        examples = examples or []  # ensure it's a list if empty
        prompt = chat_prompt_with_structured_output(
            parser=parser,
            preamble=preamble,
            question=question,
            instructions=instructions,
            input_variables=input_variables,
            format_instructions_variable=format_instructions_variable,
            examples=examples,
            example_input_vars=example_input_vars,
            example_output_var=example_output_var,
            example_indent=example_indent,
            jinja=jinja,
            use_system_prompt=use_system_prompt,
        )
        return cls(
            prompt=prompt,
            llm=llm,
            memory=memory,
            output_key=output_key,
            answer_key=answer_key,
            summarized_notes_key=summarized_notes_key,
            patient_history_key=patient_history_key,
            parsed_answer_key=parsed_answer_key,
            label_mapping=label_mapping,
        )

    def _validate_evidence(
        self, inputs: Dict[str, Any], answer: PatientHistoryGroundedAnswer
    ) -> PatientHistoryGroundedAnswer:
        """Evidence parsing/validation.  This only logs a warning if the
        evidence cannot be found. If evidence is found, add begin/end span. If
        summarized notes are available we attempt to ground against the note, otherwise
        we just ground against the summarized patient history.
        """
        if self.summarized_notes_key:
            summarized_notes = inputs[self.summarized_notes_key]
            answer = validate_evidence_from_summarized_notes(answer, summarized_notes)
        elif self.patient_history_key:
            summary_text = inputs[self.patient_history_key]
            answer = validate_evidence_from_summary_text(answer, summary_text)
        return answer

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        response = self.generate([inputs], run_manager=run_manager)
        if len(response.generations) != 1:
            raise ValueError(
                "Expected a single list of Generations, got "
                f"{len(response.generations)}"
            )
        if len(response.generations[0]) != 1:
            raise ValueError(
                f"Expected a single Generation, got {len(response.generations[0])} "
                "(was the 'n' parameter > 1?)"
            )
        raw_response = response.generations[0][0].text
        answer = cast(PatientHistoryGroundedAnswer, self.parser.parse(raw_response))
        if self.normalized_label_mapping:
            answer.answer = parse_output(self.normalized_label_mapping, answer.answer)
        answer = self._validate_evidence(inputs, answer)
        output = {
            self.output_key: raw_response,
            self.answer_key: answer.answer,
            self.parsed_answer_key: answer,
        }
        return output
