import json
from typing import Dict, List

from langchain.chains.base import Chain


def _format(evidence: str, example_json: str) -> str:
    example = json.loads(example_json)
    formatted_example = "Evidence:\n"
    formatted_example += evidence
    formatted_example += "Reasoning:\n" + example["reasoning"] + "\n"
    formatted_example += f"Answer:\n{example['answer']}"

    return formatted_example


class PatientHistoryGroundedAnswerExampleFormatter(Chain):
    """Chain that takes in formatted PatientHistoryGrounedAnswer examples and returns a
    string with formatted examples."""

    input_key: str = "examples"
    output_key: str = "formatted_examples"
    example_evidence_key: str = "patient_history"
    example_result_key: str = "result"

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run the logic of this chain and return the output."""
        examples: List[dict] = inputs[self.input_key]
        output = "\n\n-----\n\n".join(
            [
                _format(
                    example[self.example_evidence_key], example[self.example_result_key]
                )
                for example in examples
            ]
        )
        return {self.output_key: output}
