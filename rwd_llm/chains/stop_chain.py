from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import LLMChain


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
