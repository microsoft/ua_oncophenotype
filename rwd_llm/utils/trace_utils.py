import logging

logger = logging.getLogger(__name__)


def get_by_path(obj, path):
    if not path:
        return obj
    if isinstance(path, str):
        path = path.split(".")
    cur_key = path[0]
    remaining_path = path[1:]
    if not isinstance(obj, (dict, list)):
        logger.warning(f"Expected object to be dict or list, found {type(obj)}")
    if isinstance(obj, list):
        cur_key = int(cur_key)
    return get_by_path(obj[cur_key], remaining_path)


def print_trace(run, indent_level=0, cidx=0):
    indent = "  " * indent_level
    next_indent = "  " * (indent_level + 1)
    name = run["name"]
    children = run["child_runs"]
    inputs = list(run["inputs"].keys())
    outputs = list(run["outputs"].keys())
    print(f"{indent}{cidx} - {name}:")
    print(f"{next_indent}inputs: {inputs}")
    print(f"{next_indent}outputs: {outputs}")
    print(f"{next_indent}children:")
    for cidx, child in enumerate(children):
        print_trace(child, indent_level + 1, cidx)
