from typing import Optional

from langchain.prompts import PromptTemplate
from langchain.schema import BaseMemory

from .concatenate_documents import ConcatenateDocuments

BASIC_NOTE_FMT = """
Note Type: {type}
Note ID: {doc_idx}
Date: {date}
----------
{page_content}
----------
"""

BASIC_FMT_INPUT_VARIABLES = ["type", "doc_idx", "date", "page_content"]

NOTE_SEP = "\n\n----------------------------------------\n\n"


def get_simple_concatenate_notes(
    input_key: str = "input_documents",
    output_key: str = "output_text",
    memory: Optional[BaseMemory] = None,
) -> ConcatenateDocuments:
    template = PromptTemplate(
        template=BASIC_NOTE_FMT, input_variables=BASIC_FMT_INPUT_VARIABLES
    )
    return ConcatenateDocuments(
        document_prompt=template,
        document_separator=NOTE_SEP,
        doc_idx_key="doc_idx",
        input_key=input_key,
        output_key=output_key,
        memory=memory,
    )
