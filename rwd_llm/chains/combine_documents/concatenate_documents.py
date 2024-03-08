from typing import Any, List, Optional, Tuple

from langchain.callbacks.manager import Callbacks
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate
from langchain.pydantic_v1 import Extra, Field
from langchain.schema import format_document
from langchain.schema.prompt_template import BasePromptTemplate


def _get_default_document_prompt() -> PromptTemplate:
    return PromptTemplate(input_variables=["page_content"], template="{page_content}")


class ConcatenateDocuments(BaseCombineDocumentsChain):
    """Basic chain that combines documents by joining them together."""

    """LLM wrapper to use after formatting documents."""
    document_prompt: BasePromptTemplate = Field(
        default_factory=_get_default_document_prompt
    )
    """Prompt to use to format each document."""
    document_separator: str = "\n\n"
    """The string with which to join the formatted documents"""
    doc_idx_key: str = "doc_idx"
    """The key to use in the metadata to store the document index."""
    sort_by: Optional[str] = None
    """Optional metadata field to sort the documents by"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def _combine_docs(self, docs: List[Document]) -> str:
        if self.sort_by is not None:
            docs = list(sorted(docs, key=lambda doc: doc.metadata[self.sort_by]))
        # Format each document according to the prompt
        for didx, doc in enumerate(docs):
            doc.metadata[self.doc_idx_key] = didx
        doc_strings = [format_document(doc, self.document_prompt) for doc in docs]
        # Join the documents together to put them in the prompt.
        joined_text = self.document_separator.join(doc_strings)
        return joined_text

    def combine_docs(
        self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        joined_text = self._combine_docs(docs)
        return joined_text, {}

    async def acombine_docs(
        self, docs: List[Document], callbacks: Callbacks = None, **kwargs: Any
    ) -> Tuple[str, dict]:
        """Stuff all documents into one prompt and pass to LLM."""
        joined_text = self._combine_docs(docs)
        return joined_text, {}

    @property
    def _chain_type(self) -> str:
        return "combine_clinical_notes"
