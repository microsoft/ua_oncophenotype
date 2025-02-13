from typing import Any, Callable, Dict, List, Optional, Union

from langchain.chains.base import Chain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.documents import Document

from ..map_documents import MapDocumentsChain


class MapDocumentsTestChain(Chain):
    mapping_func: Union[Callable[[Document], Document], Callable[[str], str]]
    input_key: str = "doc_in"
    output_key: str = "doc_out"
    text_only: bool = False

    @property
    def input_keys(self) -> List[str]:
        """Input keys this chain expects."""
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        doc = inputs[self.input_key]
        return {self.output_key: self.mapping_func(doc)}


def test_normal_usage():
    def test_doc_mapping(doc: Document) -> Document:
        return Document(
            page_content=doc.metadata["doc_id"],
        )

    # this gets applied to each doc separately
    the_chain = MapDocumentsTestChain(
        mapping_func=test_doc_mapping,
        input_key="the_doc_key",
        output_key="the_mapped_doc",
    )
    docs = [
        Document(page_content="foo", metadata={"doc_id": str(didx)})
        for didx in range(4)
    ]

    map_docs = MapDocumentsChain(
        doc_chain=the_chain,
        docs_input_key="the_docs_key",
        docs_output_key="the_mapped_docs",
        doc_chain_doc_input_key="the_doc_key",
        doc_chain_doc_output_key="the_mapped_doc",
        text_only=False,
    )
    result = map_docs({"the_docs_key": docs})
    mapped_docs = result["the_mapped_docs"]
    for didx, doc in enumerate(mapped_docs):
        assert doc.page_content == str(didx)


def test_text_usage():
    def test_doc_mapping(text: str) -> str:
        return "".join(reversed(text))

    # this gets applied to each doc separately
    the_chain = MapDocumentsTestChain(
        mapping_func=test_doc_mapping,
        input_key="the_doc_key",
        output_key="the_mapped_doc",
    )
    docs = [
        Document(page_content="foo", metadata={"doc_id": str(didx)})
        for didx in range(4)
    ]

    map_docs = MapDocumentsChain(
        doc_chain=the_chain,
        docs_input_key="the_docs_key",
        docs_output_key="the_mapped_docs",
        doc_chain_doc_input_key="the_doc_key",
        doc_chain_doc_output_key="the_mapped_doc",
        text_only=True,
    )
    result = map_docs({"the_docs_key": docs})
    mapped_docs = result["the_mapped_docs"]
    for didx, doc in enumerate(mapped_docs):
        assert doc.page_content == "oof"
        assert doc.metadata["doc_id"] == str(didx)


if __name__ == "__main__":
    test_normal_usage()
    test_text_usage()
