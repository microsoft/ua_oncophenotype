"""
Chains used for ablation study. Similar to PatientHistoryGroundedAnswerChain, but omits
evidence or evidence+reasonong.
"""

import logging
from typing import Any, Dict, List, Optional, Union, cast

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.llm import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field, validator
from langchain.schema import BaseMemory
from langchain_community.chat_models import ChatOpenAI
from rwd_llm.output_parsers import PydanticOutputParserWithExamples

from .categorical_chain import normalize_label_mapping, parse_output
from .question_with_structured_output import chat_prompt_with_structured_output

logger = logging.getLogger(__name__)


class PatientHistoryAnswer(BaseModel):
    reasoning: str = Field(description="Reasoning leading to the answer")
    answer: str = Field(description="The answer to the question")


DEFAULT_REASONING_EXAMPLE = PatientHistoryAnswer(
    reasoning="The step-by-step reasoning leading to the answer.",
    answer="The answer to the question.",
)

DEFAULT_REASONING_PARSER = PydanticOutputParserWithExamples(
    pydantic_object=PatientHistoryAnswer, examples=DEFAULT_REASONING_EXAMPLE
)


class AnswerChainNoEvidence(LLMChain):
    # unused for ablation chains:
    summarized_notes_key: Optional[str] = None
    """key containing summarized patient history as string"""
    patient_history_key: Optional[str] = None
    """prompt input variable to inject the summarized patient history"""
    answer_key: str = "answer"
    parsed_answer_key: str = "parsed_answer"
    label_mapping: Optional[Union[List[str], Dict[str, str]]] = None
    parser: PydanticOutputParser = DEFAULT_REASONING_PARSER

    @validator("patient_history_key", always=True)
    def _get_patient_history_key(cls, patient_history_key, values) -> str:
        # if there is only one input to the prompt, use it as the patient history key
        if patient_history_key is None:
            if len(values["prompt"].input_variables) == 1:
                patient_history_key = values["prompt"].input_variables[0]
            else:
                raise ValueError(
                    "patient_history_key must be specified if there are multiple inputs"
                    " to the prompt"
                )
        else:
            if patient_history_key not in values["prompt"].input_variables:
                raise ValueError(
                    f"patient_history_key {patient_history_key} not found in prompt"
                    " input_variables"
                )
        return patient_history_key

    @validator("label_mapping")
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
        summarized_notes_key: Optional[str] = None,  # unused
        instructions: Optional[str] = None,
        preamble: Optional[str] = None,
        format_instructions_variable: str = "format_instructions",
        memory: Optional[BaseMemory] = None,
        output_key="answer_raw",
        answer_key="answer",
        parsed_answer_key="parsed_answer",
        label_mapping: Optional[Union[List[str], Dict[str, str]]] = None,
        examples: Optional[List[dict]] = None,
        example_output_var: str = "result",
        example_indent: int = 2,
    ) -> "AnswerChainNoEvidence":
        parser: PydanticOutputParser = DEFAULT_REASONING_PARSER
        examples = examples or []  # ensure it's a list
        prompt = chat_prompt_with_structured_output(
            parser=parser,
            preamble=preamble,
            question=question,
            instructions=instructions,
            input_variables=input_variables,
            format_instructions_variable=format_instructions_variable,
            examples=examples,
            example_output_var=example_output_var,
            example_indent=example_indent,
        )
        return cls(
            prompt=prompt,
            llm=llm,
            memory=memory,
            output_key=output_key,
            answer_key=answer_key,
            patient_history_key=patient_history_key,
            parsed_answer_key=parsed_answer_key,
            label_mapping=label_mapping,
        )

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
        answer = cast(PatientHistoryAnswer, self.parser.parse(raw_response))
        if self.normalized_label_mapping:
            answer.answer = parse_output(self.normalized_label_mapping, answer.answer)
        output = {
            self.output_key: raw_response,
            self.answer_key: answer.answer,
            self.parsed_answer_key: answer,
        }
        return output


class AnswerOnlyAnswer(BaseModel):
    answer: str = Field(description="The answer to the question")


DEFAULT_ANSWER_ONLY_EXAMPLE = AnswerOnlyAnswer(
    answer="The answer to the question.",
)

DEFAULT_ANSWER_ONLY_PARSER = PydanticOutputParserWithExamples(
    pydantic_object=AnswerOnlyAnswer, examples=DEFAULT_ANSWER_ONLY_EXAMPLE
)


class AnswerOnlyChain(LLMChain):
    # unused for ablation chains:
    summarized_notes_key: Optional[str] = None
    """key containing summarized patient history as string"""
    patient_history_key: Optional[str] = None
    """prompt input variable to inject the summarized patient history"""
    answer_key: str = "answer"
    parsed_answer_key: str = "parsed_answer"
    label_mapping: Optional[Union[List[str], Dict[str, str]]] = None
    parser: PydanticOutputParser = DEFAULT_ANSWER_ONLY_PARSER

    @validator("patient_history_key", always=True)
    def _get_patient_history_key(cls, patient_history_key, values) -> str:
        # if there is only one input to the prompt, use it as the patient history key
        if patient_history_key is None:
            if len(values["prompt"].input_variables) == 1:
                patient_history_key = values["prompt"].input_variables[0]
            else:
                raise ValueError(
                    "patient_history_key must be specified if there are multiple inputs"
                    " to the prompt"
                )
        else:
            if patient_history_key not in values["prompt"].input_variables:
                raise ValueError(
                    f"patient_history_key {patient_history_key} not found in prompt"
                    " input_variables"
                )
        return patient_history_key

    @validator("label_mapping")
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
        summarized_notes_key: Optional[str] = None,  # unused
        instructions: Optional[str] = None,
        preamble: Optional[str] = None,
        format_instructions_variable: str = "format_instructions",
        memory: Optional[BaseMemory] = None,
        output_key="answer_raw",
        answer_key="answer",
        parsed_answer_key="parsed_answer",
        label_mapping: Optional[Union[List[str], Dict[str, str]]] = None,
        examples: Optional[List[dict]] = None,
        example_output_var: str = "result",
        example_indent: int = 2,
    ) -> "AnswerOnlyChain":
        parser: PydanticOutputParser = DEFAULT_ANSWER_ONLY_PARSER
        examples = examples or []  # ensure it's a list
        prompt = chat_prompt_with_structured_output(
            parser=parser,
            preamble=preamble,
            question=question,
            instructions=instructions,
            input_variables=input_variables,
            format_instructions_variable=format_instructions_variable,
            examples=examples,
            example_output_var=example_output_var,
            example_indent=example_indent,
        )
        return cls(
            prompt=prompt,
            llm=llm,
            memory=memory,
            output_key=output_key,
            answer_key=answer_key,
            patient_history_key=patient_history_key,
            parsed_answer_key=parsed_answer_key,
            label_mapping=label_mapping,
        )

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
        answer = cast(PatientHistoryAnswer, self.parser.parse(raw_response))
        if self.normalized_label_mapping:
            answer.answer = parse_output(self.normalized_label_mapping, answer.answer)
        output = {
            self.output_key: raw_response,
            self.answer_key: answer.answer,
            self.parsed_answer_key: answer,
        }
        return output
