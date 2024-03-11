from typing import Any, Dict, List

from langchain.chains.base import Chain
from langchain.pydantic_v1 import Field
from langchain_core.example_selectors.base import BaseExampleSelector


class FewShotChain(Chain):
    """Chain that uses input to generate examples for a few-shot prompt"""

    example_selector: BaseExampleSelector
    expected_inputs: List[str] = Field(default_factory=list)
    output_key: str

    @property
    def input_keys(self) -> List[str]:
        return list(*self.expected_inputs)

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        examples = self.example_selector.select_examples(inputs)
        return {self.output_key: examples}
