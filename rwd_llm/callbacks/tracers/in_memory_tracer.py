from typing import Dict, Optional

from langchain_core.tracers import BaseTracer, Run


class InMemoryTracer(BaseTracer):
    """Tracer that doesn't actually persist anything, but can be used for post-run
    inspection"""

    def __init__(self, run_id_input_key: Optional[str] = None):
        self.run_id_input_key = run_id_input_key
        self.persistent_run_map: Dict[str, Run] = {}
        self.counter = 0
        super().__init__()

    def _persist_run(self, run: Run) -> None:
        if not run.parent_run_id:
            # onlys persist the top level run
            run_id = self.counter
            if self.run_id_input_key:
                run_id = run.inputs[self.run_id_input_key]
            self.persistent_run_map[str(run_id)] = run
            self.counter += 1
