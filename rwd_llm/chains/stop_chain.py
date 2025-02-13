from typing import Any, Dict, List, Optional

from langchain.chains.llm import LLMChain
from langchain_core.callbacks import CallbackManagerForChainRun


class LLMStopChain(LLMChain):
    """Just a chain that accepts stop tokens during creation"""

    stop: List[str]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        inputs["stop"] = self.stop
        return super()._call(inputs, run_manager=run_manager)
