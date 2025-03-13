import collections
import json
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Union

from pydantic import BaseModel
from rwd_llm.data_loaders.data_loaders_base import DatasetBase
from rwd_llm.dtypes.dtypes_utils import force_to_json

logger = logging.getLogger(__name__)

JSONType = Union[Dict, List, str, int, float, bool, None]
SerializerType = Callable[[Any], JSONType]
DeserializerType = Callable[[JSONType], Any]


class NoOpSerializer:
    def __call__(self, obj: Any) -> JSONType:
        return obj


class NoOpDeserializer:
    def __call__(self, dict_ob: JSONType) -> Any:
        return dict_ob


class PersistentMemoryProviderBase:
    """Basic interface for a persistent memory provider. Also includes ability to
    register default and custom serializers and deserializers for specific keys."""

    def __init__(
        self,
        custom_serializers: Optional[Dict[str, SerializerType]] = None,
        default_serializer: Optional[SerializerType] = None,
        custom_deserializers: Optional[Dict[str, DeserializerType]] = None,
        default_deserializer: Optional[DeserializerType] = None,
    ):
        self.custom_serializers: Dict[str, SerializerType] = custom_serializers or {}
        self.default_serializer: SerializerType = default_serializer or NoOpSerializer()
        self.custom_deserializers: Dict[str, DeserializerType] = (
            custom_deserializers or {}
        )
        self.default_deserializer: DeserializerType = (
            default_deserializer or NoOpDeserializer()
        )

    @abstractmethod
    def keys(self, item_id: str) -> Set[str]:
        """Get all keys for a given item ID."""
        raise NotImplementedError

    @abstractmethod
    def add_memory(self, item_id: str, key: str, val: Any) -> None:
        """Add a memory to the persistent store."""
        raise NotImplementedError

    @abstractmethod
    def get_memory(self, item_id: str, key: str) -> Any:
        """Get a memory from the persistent store."""
        raise NotImplementedError

    @abstractmethod
    def add_memories(self, item_id: str, memories: Dict[str, Any]) -> None:
        """Add multiple memories to the persistent store."""
        raise NotImplementedError

    @abstractmethod
    def get_memories(self, item_id: str, keys: Optional[List[str]]) -> Dict[str, Any]:
        """Get multiple memories from the persistent store. If keys is None, get all
        memories."""
        raise NotImplementedError

    def log_error(self, item_id: str, key: str, err: Exception) -> None:
        """Log an error creating a specific memory"""
        logger.exception(f"Error creating memory {item_id}:{key}: {err}")

    def _get_serializer(self, key: str) -> SerializerType:
        return self.custom_serializers.get(key, self.default_serializer)

    def _get_deserializer(self, key: str) -> DeserializerType:
        # if key in self.custom_deserializers:
        #     logger.debug(
        #         f"using custom deserializer for {key}:"
        #         f" {type(self.custom_deserializers[key])}"
        #     )
        # else:
        #     logger.debug(f"using default deserializer for {key}")

        return self.custom_deserializers.get(key, self.default_deserializer)

    def _serialize_value(self, key: str, value: Any) -> JSONType:
        serializer = self._get_serializer(key)
        return serializer(value)

    def _deserialize_value(self, key: str, serialized_value: JSONType) -> Any:
        return self._get_deserializer(key)(serialized_value)


class EphemeralMemoryProvider(PersistentMemoryProviderBase):
    """Just an in-memory dictionary, not really persistent. Still serializes values,
    mostly for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memories: Dict[str, Dict[str, Any]] = collections.defaultdict(dict)

    def keys(self, item_id: str) -> Set[str]:
        return set(self.memories.get(item_id, {}).keys())

    def add_memory(self, item_id: str, key: str, val: Any) -> None:
        self.memories[item_id][key] = self._serialize_value(key, val)

    def add_memories(self, item_id: str, memories: Dict[str, Any]) -> None:
        if not memories:
            return
        memories = {
            key: self._serialize_value(key, val) for key, val in memories.items()
        }
        self.memories[item_id].update(memories)

    def get_memory(self, item_id: str, key: str) -> Any:
        return self.memories[item_id][key]

    def get_memories(self, item_id: str, keys: Optional[List[str]]) -> Dict[str, Any]:
        memories = self.memories[item_id]
        if keys is None:
            return dict(**memories)
        else:
            return {key: memories[key] for key in keys}


class ForceJsonSerializer:
    def __call__(self, obj):
        return force_to_json(obj)


class SerializedMemoryValue(BaseModel):
    item_id: str
    source_run: str
    key: str
    value: JSONType


class FileMemoryProvider(PersistentMemoryProviderBase):
    """Basic (multi)file-backed memory provider."""

    def __init__(self, run_id: str, persistence_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.run_id: str = run_id
        self.persistence_dir: Path = Path(persistence_dir)
        logger.info(f"Ensuring output memory directory exists '{self.persistence_dir}'")
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache: Dict[str, Dict[str, JSONType]] = {}

    @classmethod
    def initialize(
        cls,
        run_id: str,
        persistence_dir: str,
        input_dir: Optional[str] = None,
        input_keys: Optional[List[str]] = None,
        dataset: Optional[DatasetBase] = None,
        save_inputs: bool = False,
        custom_serializers: Optional[Dict[str, SerializerType]] = None,
        default_serializer: Optional[SerializerType] = None,
        custom_deserializers: Optional[Dict[str, DeserializerType]] = None,
        default_deserializer: Optional[DeserializerType] = None,
    ) -> "FileMemoryProvider":
        """Initialize memory from a directory of files.

        Args:
            run_id (str): Run ID for memories created from this run.
            persistence_dir (str): Directory to save memory files.
            input_dir (Optional[str], optional): Directory to load input memories from.
                Defaults to None.
            input_keys (Optional[List[str]], optional): If provided, only load memories
                with these keys. Defaults to None.
            dataset (Optional[DatasetBase], optional): Dataset to load memories from.
                Defaults to None.
            save_inputs (bool, optional): If True, save inputs to memory. Otherwise,
                only save outputs. Defaults to False.
        """
        fmp = FileMemoryProvider(
            run_id=run_id,
            persistence_dir=persistence_dir,
            custom_serializers=custom_serializers,
            default_serializer=default_serializer,
            custom_deserializers=custom_deserializers,
            default_deserializer=default_deserializer,
        )

        logger.info(f"loading memories from input dir '{input_dir}'")
        if input_dir:
            ids_to_initialize = None
            if dataset is not None:
                ids_to_initialize = set([ob.id for ob in dataset])
            logger.info(f"loading memories from {input_dir}")
            fmp._load_memories(
                input_dir=Path(input_dir),
                item_ids=ids_to_initialize,
                keys=input_keys,
            )
            if save_inputs:
                # force writing values from the cache to disk
                for item_id in fmp.memory_cache:
                    for key in fmp.memory_cache[item_id]:
                        val = fmp.get_memory(item_id, key)
                        fmp._write_memory(item_id=item_id, key=key, val=val)
        return fmp

    def _add_to_cache(self, item_id: str, key: str, val: Any) -> None:
        if item_id not in self.memory_cache:
            self.memory_cache[item_id] = {}
        self.memory_cache[item_id][key] = self._serialize_value(key, val)

    def _read_from_cache(self, item_id: str, key: str) -> Any:
        return self._deserialize_value(key, self.memory_cache[item_id][key])

    def _load_memories(
        self,
        input_dir: Path,
        item_ids: Optional[Iterable[str]] = None,
        keys: Optional[List[str]] = None,
    ):
        """Load memories from persistence_dir into the memory cache."""
        input_persistence_dir = Path(input_dir)
        logger.debug(f"loading memories from {input_dir}")
        ids_to_initialize = set()
        if item_ids is None:
            for file in input_persistence_dir.iterdir():
                if file.is_dir():
                    ids_to_initialize.add(file.name)
        else:
            ids_to_initialize.update(item_ids)

        logger.debug(f"loading memories for {len(ids_to_initialize)} items")
        for item_id in ids_to_initialize:
            item_dir: Path = input_persistence_dir / item_id
            keys_to_load = set()
            if keys is None:
                for file in item_dir.iterdir():
                    if file.suffix == ".json":
                        keys_to_load.add(file.stem)
            else:
                keys_to_load.update(keys)
            for key in keys_to_load:
                val = self._read_memory(
                    item_id, key, persistence_dir=input_persistence_dir
                )
                self._add_to_cache(item_id, key, val)

    def keys(self, item_id: str) -> Set[str]:
        """Get all keys for a given item ID."""
        return set(self.memory_cache.get(item_id, {}).keys())

    def _read_memory(
        self, item_id: str, key: str, persistence_dir: Optional[Path] = None
    ) -> Any:
        if persistence_dir is None:
            persistence_dir = self.persistence_dir
        key_file = persistence_dir / item_id / f"{key}.json"
        obj = json.loads(key_file.read_text())
        serialized_val = SerializedMemoryValue.model_validate(obj)
        return self._deserialize_value(key, serialized_val.value)

    def _write_memory(self, item_id: str, key: str, val: Any) -> None:
        serialized_val = self._serialize_value(key, val)
        # print(f"orig memory to {item_id}:{key}: {val}")
        # print(f"Adding memory to {item_id}:{key}: {serialized_val}")
        mem_val = SerializedMemoryValue(
            item_id=item_id, source_run=self.run_id, key=key, value=serialized_val
        )
        item_dir = self.persistence_dir / item_id
        if not item_dir.exists():
            item_dir.mkdir()
        key_file = item_dir / f"{key}.json"
        with open(key_file, "w") as fp:
            json.dump(mem_val.model_dump(), fp)

    def _write_error(self, item_id: str, key: str, err_msg: str) -> None:
        # we use the SerializedMemoryValue class, but we just use the string directly
        # (we don't actually 'serialize' it)
        err_val = SerializedMemoryValue(
            item_id=item_id, source_run=self.run_id, key=key, value=err_msg
        )
        item_err_dir = self.persistence_dir / "errors" / item_id
        if not item_err_dir.exists():
            item_err_dir.mkdir(parents=True)
        err_file = item_err_dir / f"{key}.json"
        with open(err_file, "w") as fp:
            json.dump(err_val.model_dump(), fp)

    def add_memory(self, item_id: str, key: str, val: Any) -> None:
        """Add a memory to the persistent store."""
        self._add_to_cache(item_id, key, val)
        self._write_memory(item_id=item_id, key=key, val=val)

    def add_memories(self, item_id: str, memories: Dict[str, Any]) -> None:
        """Add multiple memories to the persistent store."""
        for memory_key, memory_val in memories.items():
            self.add_memory(item_id, memory_key, memory_val)

    def log_error(self, item_id: str, key: str, err: Exception) -> None:
        super().log_error(item_id, key, err)
        self._write_error(item_id, key, repr(err))

    def get_memory(self, item_id: str, key: str) -> Any:
        """Get a memory from the persistent store."""
        logger.debug(f"getting memory for {item_id} {key}")
        return self._read_from_cache(item_id, key)

    def get_memories(self, item_id: str, keys: Optional[List[str]]) -> Dict[str, Any]:
        """Get multiple memories from the persistent store. If keys is None, get all
        memories."""
        if keys is None:
            keys = list(self.keys(item_id))
        return {key: self.get_memory(item_id, key) for key in keys}
