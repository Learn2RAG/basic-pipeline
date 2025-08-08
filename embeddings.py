from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np
import warnings

def create_embeddings(input_sample: List[str], model_name: str = "BAAI/bge-m3") -> Union[np.ndarray, dict]:
    if model_name == "BAAI/bge-m3":
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        embeddings = model.encode(
            input_sample,
            batch_size=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
    elif model_name == "sentence-transformers/all-mpnet-base-v2":
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        embeddings = model.encode(input_sample)
    else:
        warnings.warn(f"Embedding model unknown or not provided. Using default model: BAAI/bge-m3")
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        embeddings = model.encode(
            input_sample,
            batch_size=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

    return embeddings