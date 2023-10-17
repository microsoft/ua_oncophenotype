import os
from typing import Dict, Optional

from langchain.embeddings import OpenAIEmbeddings  # no azure specific library

from .chroma import Chroma


class IndexLoader:
    def __init__(self, index_path: str, embedding_name: str):
        self.index_path = index_path
        self.embedding_name = embedding_name

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
        returns: Chroma object
        """
        if os.path.exists(index_path):
            print("Loading index from", index_path)
        else:
            raise ValueError("Index path does not exist.")
        embeddings = IndexLoader._get_embeddings_model(embedding_name)
        index = Chroma(persist_directory=index_path, embedding_function=embeddings)
        return index


class IndexQuery:
    def __init__(
        self,
        index: Chroma,
        query: str,
        search_mode: str,
        k: int,
        fetch_k: int = 20,
        filter: Optional[Dict[str, str]] = None,
        filter_document: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Queries the index for the given query.
        @index: index to search in
        @query: query to search for
        @search_mode: mode to search in
        """
        self.index = index
        self.query = query
        self.search_mode = search_mode
        self.k = k
        self.fetch_k = fetch_k
        self.filter = filter
        self.filter_document = filter_document
        self.kwargs = kwargs

    def run(self) -> list:
        if self.search_mode == "similarity":
            return self.index.similarity_search(
                query=self.query,
                k=self.k,
                filter=self.filter,
                filter_document=self.filter_document,
                **self.kwargs
            )
        elif self.search_mode == "mmr":
            return self.index.max_marginal_relevance_search(
                query=self.query,
                k=self.k,
                fetch_k=self.fetch_k,
                filter=self.filter,
                filter_document=self.filter_document,
                **self.kwargs
            )
        else:
            raise ValueError("Invalid search mode.")


class IndexSearcher:
    def __init__(self, embedding_name: str):
        self.index_loader = IndexLoader(embedding_name)

    def load_and_query_index(
        self,
        query: str,
        index_path: str,
        search_mode: str,
        k: int,
        fetch_k: int = 20,
        filter: Optional[Dict[str, str]] = None,
        filter_document: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> list:
        """
        Queries the index for the given query.
        @query: query to search for
        @index_path: name of the index to search in
        @search_mode: mode to search in
        returns: list of tuples (document_id, score)
        """
        index = self.index_loader.load_index(index_path=index_path)
        query_obj = IndexQuery(
            index=index,
            query=query,
            search_mode=search_mode,
            k=k,
            fetch_k=fetch_k,
            filter=filter,
            filter_document=filter_document,
            **kwargs
        )

        return query_obj.run()
