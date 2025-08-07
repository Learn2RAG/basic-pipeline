from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer

def create_embeddings(input_sample: list[str], model_name):
    default_model_name = "BAAI/bge-m3"
    if model_name == "BAAI/bge-m3":
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        embeddings = model.encode(
            input_sample,
            batch_size=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )  # Todo Device (nutzt momentan alle verfügbaren GPUs)
    elif model_name == "sentence-transformers/all-mpnet-base-v2":
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        embeddings = model.encode(input_sample, convert_to_numpy=True)
    else:
        model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        print('Using default model: {}'.format(default_model_name))
        embeddings = model.encode(
            input_sample,
            batch_size=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        ) # Todo Device (nutzt momentan alle verfügbaren GPUs)
    return embeddings