import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import LLMChain
from langchain.pydantic_v1 import Field, validator
from llm_lib.chains.chain_utils import ERROR_LABEL

logger = logging.getLogger(__name__)


def normalize_label_mapping(
    label_mapping: Union[List[str], Dict[str, str]]
) -> Dict[str, str]:
    if isinstance(label_mapping, list):
        label_mapping = {label: label for label in label_mapping}
    label_mapping = {
        response.strip().lower(): label for response, label in label_mapping.items()
    }
    return label_mapping


def parse_output(label_mapping: Dict[str, str], response: str) -> str:
    # Find the first matched label in the response, prioritizing longer labels if they
    # both start at the same position.

    # find the first matched label in the response and take the longest matched one
    response = response.strip().lower()
    matched_labels: Dict[str, Tuple[int, int]] = {}
    for k, v in label_mapping.items():
        ind = response.find(k.lower())
        if ind >= 0:
            matched_labels[v] = (ind, -len(k))
    if len(matched_labels) == 0:
        logger.error(
            f"Could not parse response: {response} with mapping {label_mapping}"
        )
        return ERROR_LABEL
    else:
        return min(matched_labels, key=matched_labels.get)


class CategoricalChain(LLMChain):
    label_mapping: Union[List[str], Dict[str, str]]
    label_key: str = "label"
    stop: List[str] = Field(default_factory=list)

    @validator("label_mapping")
    def _normalize_label_mapping(cls, label_mapping):
        return normalize_label_mapping(label_mapping)

    @property
    def normalized_label_mapping(self) -> Dict[str, str]:
        if not isinstance(self.label_mapping, dict):
            raise ValueError(
                "label_mapping should have been normalized by the validator (got list,"
                " expected dict)"
            )
        return self.label_mapping

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key, self.label_key]

    def _parse_output(self, response: str) -> str:
        return parse_output(self.normalized_label_mapping, response.lower())

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        inputs["stop"] = self.stop
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
        label = self._parse_output(raw_response)
        output = {self.output_key: raw_response, self.label_key: label}
        return output
