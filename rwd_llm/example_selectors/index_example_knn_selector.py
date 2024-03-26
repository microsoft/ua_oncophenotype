import collections
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import chromadb
import openai
import pandas as pd
from chromadb.api.models.Collection import Collection
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_openai import AzureOpenAIEmbeddings
from rwd_llm.chains.patient_history_grounded_answer_chain import (
    PatientHistoryGroundedAnswer,
)
from rwd_llm.utils import load_dataframe

from .example_selector_utils import grounded_answer_memory_to_example

logger = logging.getLogger(__name__)


class IndexExampleKNNSelector(BaseExampleSelector):
    """Example selector that retrieves results of PatientHistoryGroundedAnswerChain from
    persistent memory, and formats them as few-shot examples. Also allows filtering by
    metadata retrived from a dataframe.
    """

    def __init__(
        self,
        memory_dir: Union[Path, str],
        memory_name: str,
        embedding_model_name: Union[str, Embeddings],
        query_text_input: str,
        persist_path: Optional[str] = None,
        example_metadata_path: Optional[Union[pd.DataFrame, str]] = None,
        metadata_filter_col_map: Optional[Dict[str, str]] = None,
        id_column: str = "id",
        data_root_dir: Optional[str] = None,
        filter_cols: Optional[List[str]] = None,
        example_patient_history_key: str = "patient_history",
        example_result_key: str = "result",
        example_index_key: str = "evidence",
        build_index: bool = True,
        k: int = 5,
    ):
        self.k = k
        self.id_column = id_column
        self.filter_cols = filter_cols or []
        self.example_patient_history_key = example_patient_history_key
        self.example_result_key = example_result_key
        self.query_text_input = query_text_input
        self.example_patient_history_key = example_patient_history_key
        self.example_result_key = example_result_key

        # example_index_key tells whether to build index on 'evidence' or
        # 'patient_history' loaded from the memory
        self.example_index_key = example_index_key

        # metadata_filter_col_map is a mapping from the column names in the metadata to
        # the input variables that should be used to filter the examples.
        self.metadata_filter_col_map = metadata_filter_col_map or {}

        # metadata is a dataframe that holds values for each datapoint that we'll use
        # for filtering. The index is the id_column, and the columns are values that
        # should match an input example if we're going to use this as a few-shot
        # example.
        self.metadata = None

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

        if persist_path is not None:
            # store the chromadb locally and can be later loaded without building
            # (build_index == false)
            logger.info(f"ChromaDB persistence directory: {persist_path}")
            chroma_client = chromadb.PersistentClient(path=persist_path)
        else:
            chroma_client = chromadb.Client()

        if isinstance(embedding_model_name, Embeddings):
            self.embeddings_model = embedding_model_name
        elif isinstance(embedding_model_name, str):
            self.embeddings_model = self._get_embeddings_model(embedding_model_name)
        else:
            raise ValueError(
                "embedding_model_name must be an instance of Embeddings or a string"
            )
        embedding_function = self.embeddings_model.embed_documents
        # create a new collection or load it if persist_path exists already
        self.collection: Collection = chroma_client.get_or_create_collection(
            "index_collection", embedding_function=embedding_function
        )

        if build_index:
            mem_dir = Path(memory_dir) if isinstance(memory_dir, str) else memory_dir
            self._build_index(mem_dir=mem_dir, memory_name=memory_name)
        else:
            if persist_path is None or self.collection.count() == 0:
                raise ValueError(
                    "persist_path is None or no chroma stored in the persist_path."
                )

        self.chroma = Chroma(
            client=chroma_client,
            collection_name="index_collection",
            embedding_function=self.embeddings_model,
        )

    def _build_index(self, mem_dir: Path, memory_name: str):
        logger.info(f"Building index from memory directory: {mem_dir}")
        for item_dir in mem_dir.iterdir():
            if item_dir.is_dir():
                key_file = item_dir / f"{memory_name}.json"
                if key_file.exists():
                    item_id = item_dir.name
                    if self.metadata is not None:
                        # if metadata is specified, only add examples if their
                        # corresponding metadata exist
                        try:
                            text_metadata = self.metadata.loc[item_id].to_dict()
                        except Exception:
                            continue
                    else:
                        text_metadata = {}

                    example = grounded_answer_memory_to_example(
                        memory_file=key_file,
                        example_patient_history_key=self.example_patient_history_key,
                        example_result_key=self.example_result_key,
                        id_key="id",
                        patient_id=item_id,
                    )

                    if self.example_index_key == "evidence":
                        # build chroma collection indexed by the evidence
                        result_dict = json.loads(example[self.example_result_key])
                        item_evidences = [
                            re.sub(r"([^\w\s]|_)+(?=\s|$)", "", item["evidence"]) + "."
                            for item in result_dict["evidence"]
                        ]
                        text = " ".join(item_evidences)
                    else:
                        # build chroma collection indexed by summarized patient history
                        text = example[self.example_patient_history_key]

                    text_metadata.update(example)

                    # xxxx TODO: we can add text splitter if needed,
                    # xxxx       but so far we just load the entire texts without
                    # xxxx       splitting them into chunks.
                    self.add_example(self.collection, text, text_metadata, item_id)

    @staticmethod
    def _get_embeddings_model(model_name: str) -> AzureOpenAIEmbeddings:
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=openai.azure_endpoint, model=model_name
        )
        return embeddings

    def _grounded_answer_to_example(
        self, p: PatientHistoryGroundedAnswer
    ) -> Dict[str, str]:
        patient_history = "Summarized Patient History:\n\n"
        evidence_by_note_id = collections.defaultdict(list)
        for evidence in p.evidence:
            evidence_by_note_id[evidence.note_id].append(evidence.evidence)
        for evidence in p.contradictory_evidence:
            evidence_by_note_id[evidence.note_id].append(evidence.evidence)
        for note_idx, (note_id, evidence) in enumerate(evidence_by_note_id.items()):
            patient_history += f"Note ID: {note_idx}\n----------\n"
            for e in evidence:
                patient_history += f"  - {e}\n"
        encoded_result = p.json()

        return {
            self.example_patient_history_key: patient_history,
            self.example_result_key: encoded_result,
        }

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        query_text = input_variables[self.query_text_input]
        logger.debug(f"Query text: {query_text}")

        metadata_filter = {}
        for col in self.filter_cols:
            try:
                input_value = input_variables[
                    self.metadata_filter_col_map.get(col, col)
                ]
                metadata_filter[col] = input_value
            except Exception:
                continue

        logger.debug(f"metadata_filter: {metadata_filter}")
        results: List[Document] = self.chroma.similarity_search(
            query=query_text, filter=metadata_filter, k=self.k
        )
        logger.debug(f"results: {results}")
        # print("xxxx ---------------------------------------------------")
        # print(f"xxxx query: [{input_value}, {query_text}]")
        # print("xxxx results:")
        # for doc in results:
        #     print("    ", doc)
        # print("xxxx ---------------------------------------------------")

        return [doc.metadata for doc in results]

    def add_example(
        self, collection: Collection, text: str, text_metadata: Dict[str, str], id: str
    ):
        """Add new example to collection."""
        # ChromaDB 'Document's are just strings
        collection.add(documents=[text], metadatas=[text_metadata], ids=[id])
