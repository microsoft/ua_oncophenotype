import datetime as dt
from typing import Dict, Iterable, Tuple, Union

import pandas as pd
from langchain.embeddings import OpenAIEmbeddings  # no azure specific library
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai.error import RateLimitError
from rwd_llm.data_loaders.data_loaders_base import DatasetBase
from rwd_llm.dtypes.dtypes import BaseObject, ClinicalNote, Patient
from tenacity import retry
from tenacity.retry import retry_if_exception_type
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_random_exponential
from tqdm import tqdm

from .chroma import Chroma


class BuildIndex:
    def __init__(
        self,
        dataset: DatasetBase,
        embedding_model_name: str,
        chunk_size: int,
        chunk_overlap: int,
        index_output_dir: str,
    ):
        self.dataset = dataset
        self.embeddings_model = self._get_embeddings_model(embedding_model_name)
        self.text_splitter = self._get_text_splitter(chunk_size, chunk_overlap)
        self.index_output_dir = index_output_dir
        self.index = self.build_index()

    def build_index(self):
        docs = self.dataset.get_items()
        chroma = self._create_chroma_index_from_docs(
            docs, self.embeddings_model, self.text_splitter, self.index_output_dir
        )
        chroma.persist()
        return chroma

    def get_index(self):
        return self.index

    @staticmethod
    def _get_embeddings_model(model_name: str) -> OpenAIEmbeddings:
        embeddings = OpenAIEmbeddings(model=model_name)
        return embeddings

    @staticmethod
    def _get_text_splitter(
        chunk_size: int = 1000, chunk_overlap: int = 50
    ) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            separators=["\n\n", "\n", " ", ""],
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(RateLimitError),
    )
    def _create_embedding_with_retry(chroma, chunked_text, metadata) -> None:
        chroma.add_texts([chunked_text], [metadata])

    @staticmethod
    def _create_chroma_index_from_df(
        df: pd.DataFrame,
        text_column_name: str,
        embeddings_model: Embeddings,
        text_splitter: RecursiveCharacterTextSplitter,
        index_output_dir: str,
    ) -> Chroma:
        chroma = Chroma(
            embedding_function=embeddings_model,
            persist_directory=index_output_dir,
        )
        list_of_dicts = df.to_dict(orient="records")
        for d in tqdm(list_of_dicts):
            text = d.pop(text_column_name)
            metadata = d
            text_chunks = text_splitter.split_text(text)
            for i in range(len(text_chunks)):
                BuildIndex._create_embedding_with_retry(
                    chroma, text_chunks[i], metadata
                )

        return chroma

    @staticmethod
    def _transform_doc(
        doc: BaseObject,
    ) -> Tuple[str, Dict[str, Union[str, int, float]]]:
        d = doc.to_dict()
        text = d.pop("text")
        metadata = d
        if isinstance(doc, Patient):
            metadata["patient_id"] = doc.id
        if isinstance(doc, ClinicalNote):
            metadata["note_id"] = doc.id
            metadata["timestamp"] = dt.datetime.fromisoformat(doc.date).timestamp()
        return text, metadata

    def _create_chroma_index_from_docs(
        self,
        docs: Iterable[BaseObject],
        embeddings_model: Embeddings,
        text_splitter: RecursiveCharacterTextSplitter,
        index_output_dir: str,
    ) -> Chroma:
        chroma = Chroma(
            embedding_function=embeddings_model,
            persist_directory=index_output_dir,
        )
        for doc in tqdm(docs):
            text, metadata = self._transform_doc(doc)
            d = doc.to_dict()
            text = d.pop("text")
            metadata = d
            text_chunks = text_splitter.split_text(text)
            # REMOVEME
            print(f"creating embedding with metadata {metadata}")
            # END REMOVEME
            for i in range(len(text_chunks)):
                BuildIndex._create_embedding_with_retry(
                    chroma, text_chunks[i], metadata
                )

        return chroma
