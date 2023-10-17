import collections
import logging
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.loading import type_to_loader_dict
from langchain.llms.loading import load_llm, load_llm_from_config
from langchain.prompts.loading import load_prompt, load_prompt_from_config
from langchain.schema import Generation
from rwd_llm.chains.categorical_chain import CategoricalChain

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


def _load_consistency_chain(config: dict, **kwargs: Any) -> LLMConsistencyChain:
    if "llm" in config:
        llm_config = config.pop("llm")
        llm = load_llm_from_config(llm_config)
    elif "llm_path" in config:
        llm = load_llm(config.pop("llm_path"))
    else:
        raise ValueError("One of `llm` or `llm_path` must be present.")
    if "prompt" in config:
        prompt_config = config.pop("prompt")
        prompt = load_prompt_from_config(prompt_config)
    elif "prompt_path" in config:
        prompt = load_prompt(config.pop("prompt_path"))
    else:
        raise ValueError("One of `prompt` or `prompt_path` must be present.")

    if "stop" not in config:
        raise ValueError("Must specify `stop`.")
    stop = config.pop("stop")

    return LLMConsistencyChain(
        stop=stop,
        prompt=prompt,
        llm=llm,
        **config,
    )


# hack to register the chain type
type_to_loader_dict[LLM_CONSISTENCY_CHAIN_TYPE] = _load_consistency_chain
