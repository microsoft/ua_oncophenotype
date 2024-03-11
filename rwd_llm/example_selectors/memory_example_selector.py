import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.example_selectors.base import BaseExampleSelector
import pandas as pd

from rwd_llm.memory.persistent_kv_memory import SerializedMemoryValue
from rwd_llm.memory import PickleDeserializer
from rwd_llm.utils import load_dataframe

DESERIALIZER = PickleDeserializer()


class MemoryExampleSelector(BaseExampleSelector):
    """Example selector that retrieves examples from persistent memory. Also allows
    filtering by metadata retrived from a dataframe."""

    def __init__(
        self,
        memory_dir: Union[Path, str],
        memory_name: str,
        n_examples: int,
        example_metadata_path: Optional[Union[pd.DataFrame, str]] = None,
        id_column: str = "id",
        data_root_dir: Optional[str] = None,
        filter_cols: Optional[List[str]] = None,
    ):
        self.n_examples = n_examples
        self.id_column = id_column
        self.filter_cols = filter_cols or []
        self.metadata = None
        # metadata is a dataframe that holds values for each datapoint that we'll use
        # for filtering. The index is the id_column, and the columns are values that
        # should match an input example if we're going to use this as a few-shot
        # example.
        if isinstance(example_metadata_path, (str, Path)):
            metadata_df = load_dataframe(
                example_metadata_path, data_root_dir=data_root_dir
            )
        elif isinstance(example_metadata_path, pd.DataFrame):
            metadata_df = example_metadata_path
        elif example_metadata_path is None:
            metadata_df = None
        else:
            raise ValueError(
                "example_metadata_path must be a path to a dataframe or a dataframe"
            )
        if metadata_df is not None:
            self.metadata = metadata_df.set_index(id_column)[filter_cols]

        # Examples are the memories we'll use as examples, indexed by item_id
        mem_dir = Path(memory_dir) if isinstance(memory_dir, str) else memory_dir
        self.examples = {}
        for item_dir in mem_dir.iterdir():
            if item_dir.is_dir():
                key_file = item_dir / f"{memory_name}.json"
                if key_file.exists():
                    item_id = item_dir.name
                    obj = json.loads(key_file.read_text())
                    memory = SerializedMemoryValue.parse_obj(obj)
                    value = DESERIALIZER(memory.value)
                    self.examples[str(item_id)] = value

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        example_ids = []
        if self.metadata is not None:
            for item_id, example in self.metadata.iterrows():
                if all(
                    input_variables[col] == example[col] for col in self.filter_cols
                ):
                    example_ids.append(item_id)
                if len(example_ids) >= self.n_examples:
                    break
        else:
            example_ids = list(self.examples.keys())[: self.n_examples]
        return [self.examples[ex_id] for ex_id in example_ids]

    def add_example(self, example: Dict[str, str]) -> Any:
        """Add new example to store."""
        self.examples[example[self.id_column]] = example
