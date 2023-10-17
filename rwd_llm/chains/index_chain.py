from typing import Any, Dict, List, Optional, Type, Union

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.loading import (
    load_chain,
    load_chain_from_config,
    type_to_loader_dict,
)
from langchain.vectorstores.base import VectorStore

from ..retrieval.index_wrapper import IndexWrapper
from .chat_evidence_chain import ChatEvidenceChain
from .evidence_chain import EvidenceChain
from .evidence_chain_prompts import EvidenceChainExample

INDEX_CHAIN_TYPE = "llm_index_chain"


class IndexChain(Chain):
    llm_chain: Union[EvidenceChain, ChatEvidenceChain]
    index: VectorStore
    query: str
    search_mode: str
    k: int
    # Either a string indicating the input key to filter by, or a dict mapping the input
    # key to the filter key. For instance, when filtering by patient_id, the input key
    # is "id" for a patient object, but must be mapped to "patient_id" in the index
    # data.
    filter_by: Optional[Union[str, Dict[str, str]]] = None

    @classmethod
    def from_openai_llm(
        cls,
        index: VectorStore,
        preamble: str,
        question: str,
        llm_class: Type[Any],
        answer_options: Union[List[str], Dict[str, str]],
        llm_extra_args: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        search_mode: str = "similarity",
        k: int = 5,
        chat_model: Optional[bool] = False,
        filter_by: Optional[Union[str, Dict[str, str]]] = None,
        examples: List[EvidenceChainExample] = [],
        consistency_temperature: float = 0.7,
        n: int = 20,
        **kwargs: Any,
    ) -> "IndexChain":
        """Create a chain from an OpenAI LLM class"""

        if query is None:
            query = question

        if chat_model:
            return cls(
                llm_chain=ChatEvidenceChain.from_openai_llm(
                    preamble=preamble,
                    question=question,
                    llm_class=llm_class,
                    llm_extra_args=llm_extra_args,
                    examples=examples,
                    answer_mapping=answer_options,
                    temperature=consistency_temperature,
                    n=n,
                ),
                index=index,
                query=query,
                search_mode=search_mode,
                k=k,
                filter_by=filter_by,
                **kwargs,
            )

        return cls(
            llm_chain=EvidenceChain.from_openai_llm(
                preamble=preamble,
                question=question,
                llm_class=llm_class,
                llm_extra_args=llm_extra_args,
                examples=examples,
                answer_mapping=answer_options,
                consistency_temperature=consistency_temperature,
                num_consistency_completions=n,
            ),
            index=index,
            query=query,
            search_mode=search_mode,
            k=k,
            filter_by=filter_by,
            **kwargs,
        )

    @property
    def input_keys(self) -> List[str]:
        """Defines the input keys."""
        if self.filter_by is None:
            return []
        elif isinstance(self.filter_by, str):
            return [self.filter_by]
        elif isinstance(self.filter_by, dict):
            return list(self.filter_by.keys())
        else:
            raise ValueError(f"filter_by must be str or dict, got {self.filter_by}")

    @property
    def output_keys(self) -> List[str]:
        """Defines the output keys."""
        return self.llm_chain.output_keys + ["docs"]

    @property
    def _chain_type(self) -> str:
        return INDEX_CHAIN_TYPE

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        if "query" not in inputs:
            inputs["query"] = self.query
        if "search_mode" not in inputs:
            inputs["search_mode"] = self.search_mode
        if "k" not in inputs:
            inputs["k"] = self.k

        filter = None
        if self.filter_by is not None:
            if isinstance(self.filter_by, str):
                filter = {self.filter_by: inputs[self.filter_by]}
            elif isinstance(self.filter_by, dict):
                if len(self.filter_by) != 1:
                    raise ValueError(
                        "filter_by dict must have exactly one key-value pair"
                    )
                filter = {v: inputs[k] for k, v in self.filter_by.items()}
            else:
                raise ValueError(f"filter_by must be str or dict, got {self.filter_by}")

        docs = IndexWrapper.query(index=self.index, filter=filter, **inputs)
        doc_list = set([doc.page_content for doc in docs])
        doc_text = "\n\n".join([doc for doc in doc_list])
        llm_inputs = {**inputs, **{"text": doc_text}}

        callbacks = _run_manager.get_child()
        llm_outputs = self.llm_chain(
            llm_inputs, return_only_outputs=True, callbacks=callbacks
        )
        return {**llm_outputs, **{"docs": docs}}


def _load_index_chain(config: dict, **kwargs: Any) -> IndexChain:
    if "llm_chain" in config:
        llm_chain_config = config.pop("llm_chain")
        llm_chain = load_chain_from_config(llm_chain_config)
    elif "llm_chain_path" in config:
        llm_chain = load_chain(config.pop("llm_chain_path"))
    else:
        raise ValueError("One of `llm_chain` or `llm_chain_path` must be present.")
    if "index" in config:
        index = config.pop("index")
    else:
        raise ValueError(" index must be present.")
    if "query" in config:
        query = config.pop("query")
    else:
        query = None
    if "k" in config:
        k = config.pop("k")
    else:
        k = 5
    if "filter_by" in config:
        filter_by = config.pop("filter_by")
    else:
        filter_by = None
    if "search_mode" in config:
        search_mode = config.pop("search_mode")
    else:
        search_mode = "similarity"
    return IndexChain(
        llm_chain=llm_chain,
        index=index,
        query=query,
        k=k,
        search_mode=search_mode,
        filter_by=filter_by,
        **config,
    )


# hack to register the chain type
type_to_loader_dict[INDEX_CHAIN_TYPE] = _load_index_chain
