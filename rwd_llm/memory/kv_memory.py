import threading
from typing import Any, Callable, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, PrivateAttr
from langchain.schema import BaseMemory


class KVMemory(BaseMemory, BaseModel):
    """Simple memory for storing outputs from chains for future use."""

    # per-thread dictionary of memories
    _thread_local_memories: threading.local = PrivateAttr(
        default_factory=threading.local
    )
    new_memory_callback: Optional[Callable[[str, Any], None]] = None
    keys: Optional[List[str]] = None

    @property
    def memories(self) -> Dict[str, str]:
        if not hasattr(self._thread_local_memories, "memories"):
            self._thread_local_memories.memories = dict()
        return self._thread_local_memories.memories

    @property
    def memory_variables(self) -> List[str]:
        return list(self.memories.keys())

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        return self.memories

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save keys specified in 'keys', or everthing if 'keys' is None."""
        if self.keys is not None:
            for out_key in outputs:
                if out_key in self.keys:
                    # store
                    self.memories[out_key] = outputs[out_key]
                    # run callback
                    if self.new_memory_callback:
                        self.new_memory_callback(out_key, outputs[out_key])

    def clear(self) -> None:
        """Delete saved key/value pairs"""
        self._thread_local_memories.memories = dict()
