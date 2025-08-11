from uuid import uuid4
import hashlib
from typing import Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant import Qdrant
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

import loaders
from embeddings import create_embeddings


def point_exists(qdrant: Qdrant, collection_name: str, path: str, chunk_hash:str) -> bool:
    filter = Filter(
        must=[
            FieldCondition(key="path", match=MatchValue(value=path)),
            FieldCondition(key="content_hash", match=MatchValue(value=chunk_hash)),
        ]
    )
    result, _ = qdrant.client.scroll(
        collection_name=collection_name, scroll_filter=filter, limit=1
    )
    return len(result) > 0


def insert(qdrant: Qdrant, collection_name: str, sample: Dict):
    point = PointStruct(
        id=uuid4().hex,
        vector={
            "dense": sample["dense_vec"],
            # "sparse": SparseVector(
            #     indices=[int(x) for x in sample["sparse_vec"].keys()],
            #     values=sample["sparse_vec"].values(),
            # ),
            # "colbert": sample["colbert_vec"],
        },
        payload={
            "content": sample["page_content"],
            "path": sample["metadata"]["source"],
            "content_hash": sample["chunk_hash"],
        },
    )
    qdrant.client.upsert(collection_name=collection_name, wait=True, points=[point])


def index(user_config, opt_config):
    # TODO: enable list of file paths in loader and adapt user_config
    # Load the documents from pdf
    all_documents = loaders.sync_pdf_loader(user_config["file_path"])
    # TODO: use ifdt loader to load pdf in json, then:
    # all_documents = loaders.json_loader("loaded_documents.json")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=opt_config["chunk_size"], chunk_overlap=opt_config["chunk_overlap"]
    )
    chunks = text_splitter.split_documents(all_documents)

    collection_name = user_config["collection_name"]

    # Init vector store
    qdrant = Qdrant(
        collection_name=collection_name,
        vector_size=opt_config["vector_size"][opt_config["embedding_model"]],
    )


    chunks_content = [chunk.page_content for chunk in chunks]
    chunk_hash = [hashlib.md5(chunk.page_content.encode()).hexdigest() for chunk in chunks]
    # Todo: handle different vector lengths for batch encoding when using sparse vectors

    embeddings = create_embeddings(chunks_content, opt_config["embedding_model"])


    if isinstance(embeddings, dict) and "dense_vecs" in embeddings:
        chunks_with_embeddings = [
            dict(chunk) | {"dense_vec": dense, "chunk_hash": c_hash}
            for chunk, dense, c_hash in zip(chunks, embeddings["dense_vecs"], chunk_hash)
        ]
    else:
        chunks_with_embeddings = [
            dict(chunk) | {"dense_vec": dense, "chunk_hash": c_hash}
            for chunk, dense, c_hash in zip(chunks, embeddings, chunk_hash)
        ]

    for sample in chunks_with_embeddings:
        if not point_exists(qdrant, collection_name, sample['metadata']['source'], sample['chunk_hash']):
            insert(qdrant, collection_name, sample)
