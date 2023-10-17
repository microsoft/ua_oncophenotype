from typing import Any, Dict, List

from langchain.chains.base import Chain


class FixedOutputChain(Chain):
    """Chain that just returns a fixed mapping from input to output."""

    output: Dict[str, Any]

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return list(self.output.keys())

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run the logic of this chain and return the output."""
        return dict(**self.output)
