import collections
import logging
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.outputs import Generation
from ua_oncophenotype.chains.categorical_chain import CategoricalChain

logger = logging.getLogger(__name__)


LLM_CONSISTENCY_CHAIN_TYPE = "llm_consistency_chain"


class LLMConsistencyChain(CategoricalChain):
    """Chain that returns self-consistency based confidence as well as text"""

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key, self.label_key, "confidence", "probabilities"]

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
        output = self._parse_response(response.generations[0])
        return output

    def _parse_response(self, responses: List[Generation]) -> Dict[str, str]:
        labels = [self._parse_output(choice.text) for choice in responses]
        label_counts = collections.Counter(labels).most_common()
        probabilities = {label: count / len(labels) for label, count in label_counts}
        probabilities = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        label, confidence = probabilities[0]
        if len(probabilities) > 1:
            confidence -= probabilities[1][1]
        probabilities = collections.OrderedDict(probabilities)
        outputs = {
            self.output_key: label,
            self.label_key: label,
            "confidence": confidence,
            "probabilities": probabilities,
        }
        return outputs

    async def _acall(self, inputs: Dict[str, str]) -> Dict[str, str]:
        raise NotImplementedError("async calls not supported")

    @property
    def _chain_type(self) -> str:
        return LLM_CONSISTENCY_CHAIN_TYPE
