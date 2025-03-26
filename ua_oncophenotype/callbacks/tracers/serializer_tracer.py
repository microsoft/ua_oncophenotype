import json
import logging
import os
from typing import Optional

from langchain_core.tracers import BaseTracer, Run
from ua_oncophenotype.dtypes import force_to_json

logger = logging.getLogger(__name__)


TRACE_DIR_NAME = "traces"


class SerializerTracer(BaseTracer):
    """Tracer that writes top-level runs to disk"""

    def __init__(
        self,
        out_dir: str,
        run_id_input_key: Optional[str] = None,
        allow_overwrite=False,
    ):
        self.out_dir = os.path.join(out_dir, TRACE_DIR_NAME)
        self.run_id_input_key = run_id_input_key
        self.allow_overwrite = allow_overwrite

        if os.path.isdir(self.out_dir):
            if not self.allow_overwrite:
                raise ValueError(f"Out dir {self.out_dir} already exists")
            else:
                logger.warning(f"Out dir {self.out_dir} already exists, overwriting")
        os.makedirs(self.out_dir, exist_ok=True)
        self.counter = 0
        super().__init__()

    def _persist_run(self, run: Run) -> None:
        if not run.parent_run_id:
            # onlys persist the top level run
            run_id = self.counter
            if self.run_id_input_key:
                run_id = run.inputs[self.run_id_input_key]

            run_path = os.path.join(self.out_dir, f"{run_id}.json")
            with open(run_path, "w") as fp:
                json.dump(force_to_json(run), fp)

            self.counter += 1
