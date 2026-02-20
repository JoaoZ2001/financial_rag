
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import VECTOR_DBS_DIR, EMBEDDING_MODEL_NAME

def get_embedding_model():
    """Returns the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def get_vector_db(company_name: str, embedding_model):
    """Returns the Chroma vector database for a specific company."""
    persist_directory = str(VECTOR_DBS_DIR / company_name.lower())
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

def get_retriever(vector_db, k=3):
    """Returns a retriever for the given vector database."""
    return vector_db.as_retriever(search_kwargs={"k": k})

# Initialize resources (singleton pattern for module-level access if needed)
embedding_model = get_embedding_model()

vector_db_amd = get_vector_db("amd", embedding_model)
vector_db_intel = get_vector_db("intel", embedding_model)
vector_db_nvidia = get_vector_db("nvidia", embedding_model)

retriever_amd = get_retriever(vector_db_amd)
retriever_intel = get_retriever(vector_db_intel)
retriever_nvidia = get_retriever(vector_db_nvidia)
