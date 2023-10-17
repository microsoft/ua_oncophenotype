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
