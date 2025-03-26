import os

from ua_oncophenotype.data_loaders.data_loaders_base import DatasetBase

from .build_index import BuildIndex
from .chroma import Chroma
from .index_wrapper import IndexWrapper


def build_and_get_index(
    dataset: DatasetBase,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    index_output_dir: str,
) -> Chroma:
    build_index = BuildIndex(
        dataset, embedding_model_name, chunk_size, chunk_overlap, index_output_dir
    )
    return build_index.get_index()


def get_existing_index(index_path: str, embedding_name: str) -> Chroma:
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
    index_wrapper = IndexWrapper(index_path, embedding_name)
    return index_wrapper.index
