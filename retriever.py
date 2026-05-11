import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient


VECTORSTORE     = "vectorstore/qdrant"
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "eafit_docs"
LLM_MODEL       = "llama3.2"
TOP_K           = 4


PROMPT_TEMPLATE = """
Eres un asistente universitario de EAFIT. Responde únicamente basándote 
en el contexto proporcionado. Si no encuentras la respuesta en el contexto, 
di claramente que no tienes esa información en los documentos disponibles.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )
    client = QdrantClient(path=VECTORSTORE)
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    print(" Vector store cargado")
    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vectorstore):
    llm = OllamaLLM(model=LLM_MODEL)
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    print(" Cadena RAG lista")
    return chain, retriever


def query(chain, retriever, question: str):
    print(f"\nPregunta: {question}")
    print("─" * 50)
    response = chain.invoke(question)
    print(f"Respuesta:\n{response}")
    print("\nFuentes utilizadas:")
    sources = []
    docs = retriever.invoke(question)
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "desconocido"))
        page   = doc.metadata.get("page", "?")
        print(f"  [{i}] {source} — página {page}")
        sources.append(source)
    
    return response, sources


if __name__ == "__main__":
    print("═" * 50)
    print("  Retriever — RAG EAFIT")
    print("═" * 50 + "\n")

    vectorstore        = load_vectorstore()
    chain, retriever   = build_rag_chain(vectorstore)

    query(chain, retriever, "¿De qué trata el documento?")