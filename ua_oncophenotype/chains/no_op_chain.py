from typing import Dict, List, Optional

from langchain.chains.base import Chain


class NoOpChain(Chain):
    """Chain that just returns one of its inputs as its output."""

    expected_keys: Optional[List[str]] = None
    the_key: str
    output_key: str = "label"

    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain expects."""
        return self.expected_keys or [self.the_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run the logic of this chain and return the output."""
        return {self.output_key: inputs[self.the_key]}
