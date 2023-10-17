import json
import logging
import os
import pprint
from dataclasses import dataclass
from typing import List, Optional, Union

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.base import Chain
from langchain.schema import BaseMemory
from llm_lib.data_loaders.data_loaders_base import DatasetBase
from llm_lib.dtypes import force_to_json
from llm_lib.eval.eval import Evaluation

from .data_runners import DatasetRunnerBase

logger = logging.getLogger(__name__)


pp = pprint.PrettyPrinter(indent=4)


@dataclass
class ExperimentResult:
    results: dict
    eval_metrics: dict

    def get_results_path(self, path: str):
        return os.path.join(path, "results.json")

    def get_eval_path(self, path: str):
        return os.path.join(path, "eval.json")

    def write(self, path: str):
        pp.pprint(self.results)
        pp.pprint(self.eval_metrics)

        os.makedirs(path, exist_ok=True)
        results_path = self.get_results_path(path)
        with open(results_path, "w") as fp:
            json.dump(force_to_json(self.results), fp)
        eval_path = self.get_eval_path(path)
        with open(eval_path, "w") as fp:
            json.dump(force_to_json(self.eval_metrics), fp)


@dataclass
class Experiment:
    dataset: DatasetBase
    chain: Chain
    data_runner: DatasetRunnerBase
    evaluation: Evaluation
    memories: Optional[Union[BaseMemory, List[BaseMemory]]] = None
    callbacks: Optional[List[BaseCallbackHandler]] = None

    def run(self) -> ExperimentResult:
        memories = self.memories or []
        if isinstance(memories, BaseMemory):
            memories = [memories]
        results = self.data_runner.run(
            self.dataset, self.chain, callbacks=self.callbacks, memories=memories
        )
        eval_metrics = self.evaluation.evaluate_results(self.dataset, results)
        return ExperimentResult(results, eval_metrics)
