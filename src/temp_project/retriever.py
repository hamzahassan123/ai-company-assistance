from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from functools import lru_cache

from temp_project.config import VECTOR_DB_PATH, require_gemini_api_key


@lru_cache(maxsize=1)
def load_vector_db() -> FAISS:
    """Create a retriever for the FAISS vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        task_type="RETRIEVAL_QUERY",
        api_key=require_gemini_api_key(),
    )
    if not (VECTOR_DB_PATH / "index.faiss").is_file() or not (
        VECTOR_DB_PATH / "index.pkl"
    ).is_file():
        raise FileNotFoundError(
            f"Vector database not found in {VECTOR_DB_PATH}. "
            "Run `temp-project-ingest` to build it."
        )
    return FAISS.load_local(
        str(VECTOR_DB_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve(k: int = 4):
    """Retrieve the top k most similar documents from the FAISS vector store."""
    if k < 1:
        raise ValueError("k must be at least 1")
    vectorstore = load_vector_db()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
