from typing import Any, Dict, List

from langchain.chains.base import Chain
from langchain.pydantic_v1 import Field
from llm_lib.dtypes import BasicNoteFormatter, ClinicalNote, NoteFormatterBase


class NoteFormatterChain(Chain):
    """Chain that just returns a fixed mapping from input to output."""

    note_formatter: NoteFormatterBase = Field(default_factory=BasicNoteFormatter)
    input_key: str = "notes"
    output_key: str = "text"

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Run the logic of this chain and return the output."""
        notes: List[ClinicalNote] = inputs[self.input_key]
        return {self.output_key: self.note_formatter.format(notes)}
