from typing import Any, Dict, List, Optional

import pandas as pd
from langchain.chains.base import Chain
from pydantic import Field

from ..utils import load_dataframe


class MappingChain(Chain):
    """Chain that just returns a fixed mapping from input to output."""

    mapping: Dict[str, Any] = Field(default_factory=dict)
    input_key: str = "label"
    output_key: str = "label"
    default: Optional[str] = None

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        """Run the logic of this chain and return the output."""
        key = inputs[self.input_key]
        if self.default is not None:
            value = self.mapping.get(key, self.default)
        else:
            value = self.mapping[key]
        return {self.output_key: value}


class DataframeMappingChain(MappingChain):
    dataframe: str
    input_col: str
    output_col: str
    data_root_dir: Optional[str] = None

    def __init__(self, **kwargs):
        dataframe_file = kwargs["dataframe"]
        if isinstance(dataframe_file, pd.DataFrame):
            # for testing, allow directly passing in dataframe
            df = dataframe_file
        else:
            data_root_dir = kwargs["data_root_dir"]
            df = load_dataframe(dataframe_file, data_root_dir=data_root_dir)
        input_col = kwargs["input_col"]
        output_col = kwargs["output_col"]
        mapping = df.set_index(input_col)[output_col].to_dict()
        kwargs["mapping"] = mapping
        super().__init__(**kwargs)
