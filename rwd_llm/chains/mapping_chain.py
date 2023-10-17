from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.pydantic_v1 import Field


class MappingChain(Chain):
    """Chain that just returns a fixed mapping from input to output."""

    mapping: Dict[str, Any] = Field(default_factory=dict)
    input_key: str = "label"
    output_key: str = "label"
    default: Optional[str] = None

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run the logic of this chain and return the output."""
        key = inputs[self.input_key]
        if self.default is not None:
            value = self.mapping.get(key, self.default)
        else:
            value = self.mapping[key]
        return {self.output_key: value}
