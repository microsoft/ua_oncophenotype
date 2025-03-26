from typing import Any, Dict, List

from langchain_core.example_selectors.base import BaseExampleSelector


class InputExampleSelector(BaseExampleSelector):
    """Very simple selector that just returns one of the inputs that should contain a
    list of examples."""

    def __init__(self, examples_input_key: str):
        self.examples_input_key = examples_input_key

    def select_examples(self, input_variables: Dict[str, Any]) -> List[dict]:
        return input_variables[self.examples_input_key]

    def add_example(self, example: Dict[str, str]) -> Any:
        """Add new example to store."""
        raise NotImplementedError(
            "This example selector does not support adding examples."
        )
