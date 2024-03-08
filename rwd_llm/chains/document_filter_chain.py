from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.docstore.document import Document


class DocumentFilterChain(Chain):
    """Chain that takes in a list of documents and returns a list of documents"""

    allow_list: Optional[Dict[str, List[str]]] = None
    deny_list: Optional[Dict[str, List[str]]] = None
    input_key: str = "docs"
    output_key: str = "docs"

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _doc_matches(self, doc: Document, key: str, value_list: List[str]) -> bool:
        """Return True if the document has a metadata 'key' that is in 'value_list'"""
        val = doc.metadata.get(key, None)
        return val in value_list

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the logic of this chain and return the output."""
        docs: List[Document] = inputs[self.input_key]
        if self.deny_list is not None:
            for key, val_list in self.deny_list.items():
                docs = [
                    doc for doc in docs if not self._doc_matches(doc, key, val_list)
                ]
        if self.allow_list is not None:
            for key, val_list in self.allow_list.items():
                docs = [doc for doc in docs if self._doc_matches(doc, key, val_list)]
        return {self.output_key: docs}
