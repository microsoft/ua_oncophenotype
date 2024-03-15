import json
from pathlib import Path
from typing import Optional, Union

from ..memory.persistent_kv_memory import DeserializerType, SerializedMemoryValue
from ..memory.pickle_serializer import PickleDeserializer


def read_memory(
    memory_dir: Union[Path, str],
    item_id: str,
    key: str,
    deserializer: Optional[DeserializerType] = None,
):
    if isinstance(memory_dir, str):
        memory_dir = Path(memory_dir)
    deserializer = deserializer or PickleDeserializer()
    key_file = memory_dir / item_id / f"{key}.json"
    obj = json.loads(key_file.read_text())
    serialized_val = SerializedMemoryValue.parse_obj(obj)
    return deserializer(serialized_val.value)
