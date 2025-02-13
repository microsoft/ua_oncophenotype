from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.documents import Document
from pydantic import model_validator


class MapDocumentsChain(Chain):
    """Class that applys a chain to a sequence of documents.  This can be the first step
    in map/reduce."""

    docs_input_key: str = "docs"
    """Key in inputs that contains the list of documents to apply the chain to."""
    docs_output_key: str = "mapped_docs"
    """Key in outputs that contains the list of transformed documents."""
    doc_chain: Chain
    """Chain to apply to each document individually."""
    doc_chain_doc_input_key: Optional[str] = None
    """Key used by doc_chain for Document input. Access through doc_input_key
    property."""
    doc_chain_doc_output_key: Optional[str] = None
    """Key used by doc_chain for Document output. Access through doc_output_key
    property."""
    text_only: bool = False
    """If True, will only pass the text of the document to the chain."""

    @model_validator(mode="before")
    def find_doc_input_key(cls, data: Dict[str, Any]) -> dict:
        doc_chain_doc_input_key = data.get("doc_chain_doc_input_key")
        if doc_chain_doc_input_key is None:
            doc_chain = data["doc_chain"]
            if len(doc_chain.input_keys) > 1:
                raise ValueError(
                    "doc_chain has multiple input keys, please specify doc_input_key:"
                    f" {doc_chain.input_keys}"
                )
            key = doc_chain.input_keys[0]
        else:
            key = doc_chain_doc_input_key
        data["doc_chain_doc_input_key"] = key
        return data

    @property
    def doc_input_key(self) -> str:
        if self.doc_chain_doc_input_key is None:
            raise ValueError(
                "doc_chain_doc_input_key should have been set by validator!"
            )
        return self.doc_chain_doc_input_key

    @model_validator(mode="before")
    def find_doc_output_key(cls, data: Dict[str, Any]) -> dict:
        doc_chain_doc_output_key = data.get("doc_chain_doc_output_key")
        if doc_chain_doc_output_key is None:
            doc_chain = data["doc_chain"]
            if len(doc_chain.output_keys) > 1:
                raise ValueError(
                    "doc_chain has multiple output keys, please specify doc_output_key:"
                    f" {doc_chain.output_keys}"
                )
            key = doc_chain.output_keys[0]
        else:
            key = doc_chain_doc_output_key
        data["doc_chain_doc_output_key"] = key
        return data

    @property
    def doc_output_key(self) -> str:
        if self.doc_chain_doc_output_key is None:
            raise ValueError(
                "doc_chain_doc_output_key should have been set by validator!"
            )
        return self.doc_chain_doc_output_key

    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain expects."""
        return [self.docs_input_key]

    @property
    def output_keys(self) -> List[str]:
        """Input keys this chain expects."""
        return [self.docs_output_key]

    def _call(
        self,
        inputs: Dict[str, List[Document]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        docs = inputs[self.docs_input_key]
        if self.text_only:
            doc_chain_inputs = [{self.doc_input_key: doc.page_content} for doc in docs]
        else:
            doc_chain_inputs = [{self.doc_input_key: doc} for doc in docs]

        callbacks = _run_manager.get_child()
        results = self.doc_chain.batch(
            # FYI - this is parallelized and so it is fast.
            doc_chain_inputs,
            callbacks=callbacks,
        )
        if self.text_only:
            out_docs = []
            for doc_idx, result in enumerate(results):
                doc = Document(
                    page_content=result[self.doc_output_key],
                    metadata=dict(**docs[doc_idx].metadata),
                )
                out_docs.append(doc)
        else:
            out_docs = [result[self.doc_output_key] for result in results]

        outputs = {self.docs_output_key: out_docs}
        return outputs
