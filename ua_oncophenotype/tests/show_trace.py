# COMMAND ----------
import json
from pathlib import Path

from ua_oncophenotype.utils import print_trace

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

print(trace["child_runs"][0]["child_runs"][0]["child_runs"][0]["inputs"]["prompts"][0])
# COMMAND ----------
print(
    trace["child_runs"][0]["child_runs"][0]["child_runs"][0]["outputs"]["generations"][
        0
    ][0]
)
print(
    trace["child_runs"][0]["child_runs"][1]["child_runs"][0]["outputs"]["generations"][
        0
    ][0]
)

# COMMAND ----------
