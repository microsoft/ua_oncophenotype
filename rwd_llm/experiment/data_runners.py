import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import tqdm
from langchain.chains.base import Chain
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.memory import BaseMemory
from langchain_core.runnables import RunnableConfig
from rwd_llm.data_loaders import DatasetBase
from rwd_llm.dtypes.dtypes import BaseObject

logger = logging.getLogger(__name__)


class DatasetRunnerBase:
    def run(
        self,
        dataset: DatasetBase,
        chain: Chain,
        memories: Optional[List[BaseMemory]] = None,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
    ) -> Dict[str, Any]:
        """Returns mapping from data element IDs to results"""
        raise NotImplementedError


@dataclass
class DatasetRunner(DatasetRunnerBase):
    n_threads: int = 1
    raise_exceptions: bool = False

    def run(
        self,
        dataset: DatasetBase,
        chain: Chain,
        memories: Optional[List[BaseMemory]] = None,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
    ) -> Dict[str, Any]:
        memories = memories or []

        def predict(ob: BaseObject):
            try:
                # reset memories between invocations
                for memory in memories:
                    memory.clear()
                args = {k: v for k, v in ob.to_dict().items() if k in chain.input_keys}
                config = RunnableConfig(callbacks=callbacks)
                r = chain.invoke(
                    args,
                    config=config,
                )
                return ob.id, r
            except Exception as e:
                print(f"Error for item_id {ob.id}: {e}")
                logger.exception(f"Error for item_id {ob.id}")
                if self.raise_exceptions:
                    raise e
                return ob.id, e

        results = {}
        if self.n_threads == 1:
            for note in tqdm.tqdm(dataset):
                results[note.id] = predict(note)[1]
        else:
            for memory in memories:
                logger.warning(
                    f"Using multhreading, ensure memory class {type(memory)} is"
                    " thread-safe"
                )
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                futures = executor.map(predict, dataset)
                results = dict(tqdm.tqdm(futures))

        return results
