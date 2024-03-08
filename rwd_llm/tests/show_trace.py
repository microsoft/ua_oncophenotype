# COMMAND ----------
import json
from pathlib import Path

from rwd_llm.utils import print_trace

# COMMAND ----------
CUR_DIR = Path(__file__).parent.absolute()
OUT_DIR = CUR_DIR / "experiment_output"
TRACE_DIR = OUT_DIR / "traces"
the_id = "pat_001"

# COMMAND ----------
trace_file = TRACE_DIR / f"{the_id}.json"
trace = json.load(trace_file.open("r"))
print_trace(trace)
# COMMAND ----------
