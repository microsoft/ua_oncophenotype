import os
from typing import Dict, Optional

from langchain.vectorstores.base import VectorStore
from langchain_openai import OpenAIEmbeddings

from .chroma import Chroma


class IndexWrapper:
    def __init__(self, index_path: str, embedding_name: str):
        self.index = self.load_index(index_path, embedding_name)

    @staticmethod
    def _get_embeddings_model(embedding_name: str) -> OpenAIEmbeddings:
        """
        Gets an OpenAIEmbeddings object.
        returns: OpenAIEmbeddings object
        """
        embeddings = OpenAIEmbeddings(
            model=embedding_name,
        )
        return embeddings

    @staticmethod
    def load_index(index_path: str, embedding_name: str) -> Chroma:
        """
        Loads the index from the given path.
        @index_path: path to the index
        @embedding_name: name of the embedding model
        returns: Chroma object
        """
        if os.path.exists(index_path):
            print("Loading index from", index_path)
        else:
            raise ValueError("Index path does not exist.")
        embeddings = IndexWrapper._get_embeddings_model(embedding_name)
        index = Chroma(persist_directory=index_path, embedding_function=embeddings)
        return index

    def _query(
        self,
        query: str,
        search_mode: str,
        k: int,
        filter: Optional[Dict[str, str]] = None,
        filter_document: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> list:
        return self.query(
            index=self.index,
            query=query,
            search_mode=search_mode,
            k=k,
            filter=filter,
            filter_document=filter_document,
            **kwargs
        )

    @staticmethod
    def query(
        index: VectorStore,
        query: str,
        search_mode: str,
        k: int,
        fetch_k: int = 20,
        filter: Optional[Dict[str, str]] = None,
        filter_document: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> list:
        if search_mode == "similarity":
            return index.similarity_search(
                query=query,
                k=k,
                filter=filter,
                filter_document=filter_document,
                **kwargs
            )
        elif search_mode == "mmr":
            return index.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                filter=filter,
                filter_document=filter_document,
                **kwargs
            )
        else:
            raise ValueError("Invalid search mode.")
