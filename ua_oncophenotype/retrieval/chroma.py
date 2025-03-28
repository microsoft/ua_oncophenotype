"""Wrapper around ChromaDB embeddings platform.
Langchain library version is missing some functionality, so we use this one instead
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type

import numpy as np
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    import chromadb
    import chromadb.config

logger = logging.getLogger(__name__)


def _results_to_docs(results: Any) -> List[Document]:
    return [doc for doc, _ in _results_to_docs_and_scores(results)]


def _results_to_docs_and_scores(results: Any) -> List[Tuple[Document, float]]:
    return [
        # TODO: Chroma can do batch querying,
        # we shouldn't hard code to the 1st result
        (Document(page_content=result[0], metadata=result[1] or {}), result[2])
        for result in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


class Chroma(VectorStore):
    """Wrapper around ChromaDB embeddings platform.

    To use, you should have the ``chromadb`` python package installed.

    Example:
        .. code-block:: python

                from langchain.vectorstores import Chroma
                from langchain.embeddings.openai import OpenAIEmbeddings

                embeddings = OpenAIEmbeddings()
                vectorstore = Chroma("langchain_store", embeddings.embed_query)
    """

    _LANGCHAIN_DEFAULT_COLLECTION_NAME = "langchain"

    def __init__(
        self,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        collection_metadata: Optional[Dict] = None,
    ) -> None:
        """Initialize with Chroma client."""
        try:
            import chromadb
            import chromadb.config
        except ImportError:
            raise ValueError(
                "Could not import chromadb python package. "
                "Please install it with `pip install chromadb`."
            )

        if client_settings:
            self._client_settings = client_settings
        else:
            self._client_settings = chromadb.config.Settings()
        self._client = chromadb.PersistentClient(
            path=persist_directory, settings=self._client_settings
        )
        self._embedding_function = embedding_function
        self._persist_directory = persist_directory
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=(
                self._embedding_function.embed_documents
                if self._embedding_function is not None
                else None
            ),
            metadata=collection_metadata,
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.

        Returns:
            List[str]: List of IDs of the added texts.
        """
        # TODO: Handle the case where the user doesn't provide ids on the Collection
        if ids is None:
            ids = [str(uuid.uuid1()) for _ in texts]
        embeddings = None
        if self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(list(texts))
        self._collection.add(
            metadatas=metadatas, embeddings=embeddings, documents=texts, ids=ids
        )
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        filter_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Run similarity search with Chroma.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Document]: List of documents most similar to the query text.
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k, filter=filter, filter_document=filter_document
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        filter_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to embedding vector.
        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
        Returns:
            List of Documents most similar to the query vector.
        """
        results = self._collection.query(
            query_embeddings=embedding,
            n_results=k,
            where=filter,
            where_document=filter_document,
        )
        return _results_to_docs(results)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,
        filter_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with Chroma with distance.

        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of documents most similar to the query
                text with distance in float.
        """
        if self._embedding_function is None:
            results = self._collection.query(
                query_texts=[query],
                n_results=k,
                where=filter,
                where_document=filter_document,
            )
        else:
            query_embedding = self._embedding_function.embed_query(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter,
                where_document=filter_document,
            )

        return _results_to_docs_and_scores(results)

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        filter: Optional[Dict[str, str]] = None,
        filter_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """

        results = self._collection.query(
            query_embeddings=embedding,
            n_results=fetch_k,
            where=filter,
            where_document=filter_document,
            include=["metadatas", "documents", "distances", "embeddings"],
        )
        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32), results["embeddings"][0], k=k
        )

        candidates = _results_to_docs(results)

        selected_results = [r for i, r in enumerate(candidates) if i in mmr_selected]
        return selected_results

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        filter: Optional[Dict[str, str]] = None,
        filter_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.
        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.
        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        if self._embedding_function is None:
            raise ValueError(
                "For MMR search, you must specify an embedding function oncreation."
            )

        embedding = self._embedding_function.embed_query(query)
        docs = self.max_marginal_relevance_search_by_vector(
            embedding, k, fetch_k, filter, filter_document
        )
        return docs

    def delete_collection(self) -> None:
        """Delete the collection."""
        self._client.delete_collection(self._collection.name)

    def persist(self) -> None:
        """
        This used to be required, but is not any longer
        """

    @classmethod
    def from_texts(
        cls: Type[Chroma],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        **kwargs: Any,
    ) -> Chroma:
        """Create a Chroma vectorstore from a raw documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            texts (List[str]): List of texts to add to the collection.
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            metadatas (Optional[List[dict]]): List of metadatas. Defaults to None.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            client_settings (Optional[chromadb.config.Settings]): Chroma client settings

        Returns:
            Chroma: Chroma vectorstore.
        """
        chroma_collection = cls(
            collection_name=collection_name,
            embedding_function=embedding,
            persist_directory=persist_directory,
            client_settings=client_settings,
        )
        chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return chroma_collection

    @classmethod
    def from_documents(
        cls: Type[Chroma],
        documents: List[Document],
        embedding: Optional[Embeddings] = None,
        ids: Optional[List[str]] = None,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        persist_directory: Optional[str] = None,
        client_settings: Optional[chromadb.config.Settings] = None,
        **kwargs: Any,
    ) -> Chroma:
        """Create a Chroma vectorstore from a list of documents.

        If a persist_directory is specified, the collection will be persisted there.
        Otherwise, the data will be ephemeral in-memory.

        Args:
            collection_name (str): Name of the collection to create.
            persist_directory (Optional[str]): Directory to persist the collection.
            ids (Optional[List[str]]): List of document IDs. Defaults to None.
            documents (List[Document]): List of documents to add to the vectorstore.
            embedding (Optional[Embeddings]): Embedding function. Defaults to None.
            client_settings (Optional[chromadb.config.Settings]): Chroma client settings
        Returns:
            Chroma: Chroma vectorstore.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return cls.from_texts(
            texts=texts,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids,
            collection_name=collection_name,
            persist_directory=persist_directory,
            client_settings=client_settings,
        )
