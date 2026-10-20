from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document 
from langchain_core.output_parsers import StrOutputParser

from temp_project.config import require_gemini_api_key
from temp_project.retriever import retrieve


llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=require_gemini_api_key(),
    temperature=0.9,
)
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Answer the user's question using only the context below. "
     "If the answer is not in the context, say you don't know.\n\n"
     "Company Context:\n{context}"),
    ("human", "\nHuman Question:\n{question}"),
])


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\n"
        f"Content: {doc.page_content}"
        for doc in docs
    )


chain = PROMPT | llm | StrOutputParser()


def ans(query: str, k: int = 3) -> str:
    """Retrieve the top k most similar documents from the FAISS vector store and answer the user's question."""
    query = query.strip()
    if not query:
        raise ValueError("Question must not be empty")

    retriever = retrieve(k=k)
    docs = retriever.invoke(query)
    if not docs:
        return "I don't have enough information in the company knowledge base."
    context = _format_docs(docs)
    answer = chain.invoke(
        {"context": context, "question": query},
        config={
            "tags": ["cogent-mind", "rag"],
            "metadata": {
                "environment": "development",
                "feature": "company-knowledge",
                "top_k": k,
            },
        },
    )
    return answer