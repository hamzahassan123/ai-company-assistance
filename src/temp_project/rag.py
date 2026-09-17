from langchain_google_genai import GoogleGenerativeAI
from temp_project.config import GEMINI_API_KEY, VECTOR_DB_PATH
from langchain_core.prompts import ChatPromptTemplate
from temp_project.retriever import retrieve
from langchain_core.documents import Document 
from langchain_core.output_parsers import StrOutputParser


llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.9,
)
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the user's question using only the context below. "
     "If the answer is not in the context, say you don't know.\n\n"
     "Company Context:\n{context}"),
    ("human", "\nHuman Question:\n{question}"),
])


def _format_docs(docs: list[Document])-> str:
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', "unknown")}\n Content: {doc.page_content}" 
        for doc in docs
        )

chain =  PROMPT | llm | StrOutputParser()

def ans(query: str, k: int = 4) -> str:
    """Retrieve the top k most similar documents from the FAISS vector store and answer the user's question."""
    retriever = retrieve(k=k)
    docs = retriever.invoke(query)
    if not docs:
        return "I don't have enough information in the company knowledge base."
    context = _format_docs(docs)
    answer = chain.invoke({"context": context, "question": query})
    return answer