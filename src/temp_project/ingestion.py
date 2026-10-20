from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from temp_project.config import (
    SALES_DATA_PATH,
    VECTOR_DB_PATH,
    require_gemini_api_key,
)


def loading_and_splitting() -> list[Document]:
    """Load Markdown sales documents and split them into embedding-sized chunks."""
    if not SALES_DATA_PATH.is_dir():
        raise FileNotFoundError(f"Sales data directory does not exist: {SALES_DATA_PATH}")

    loader = DirectoryLoader(
        str(SALES_DATA_PATH),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()
    if not documents:
        raise ValueError(f"No Markdown documents found in {SALES_DATA_PATH}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return splitter.split_documents(documents)


def vector_DB() -> FAISS:
    """Build the FAISS index, save it locally, and return it for immediate use."""
    api_key = require_gemini_api_key()
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    document_chunks = loading_and_splitting()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        task_type="RETRIEVAL_DOCUMENT",
        api_key=api_key,
    )
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    vector_store.save_local(folder_path=str(VECTOR_DB_PATH))
    return vector_store


def main() -> None:
    """Build or replace the local FAISS vector database."""
    vector_DB()
    print(f"Vector database saved to {VECTOR_DB_PATH}")
