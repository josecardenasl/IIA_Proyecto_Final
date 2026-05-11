import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient

load_dotenv()

VECTORSTORE     = "vectorstore/qdrant"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "eafit_docs"
LLM_MODEL       = "gemini-2.5-flash"
TOP_K           = 4
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")


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
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GOOGLE_API_KEY
    )
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
    docs = retriever.invoke(question)
    sources = []
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

    if not GOOGLE_API_KEY:
        raise ValueError("Falta GOOGLE_API_KEY. Agrégala al archivo .env")

    vectorstore      = load_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    query(chain, retriever, "¿Que es eafit, cuales pregrados tiene?")
