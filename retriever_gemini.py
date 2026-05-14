import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from qdrant_client import QdrantClient

load_dotenv()

DATA_DIR        = "data/"
VECTORSTORE     = "vectorstore/qdrant"
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "eafit_docs"
LLM_MODEL       = "gemini-2.5-flash"
TOP_K           = 12
CHUNK_SIZE      = 1500
CHUNK_OVERLAP   = 200
BM25_WEIGHT     = 0.5  # peso de BM25 (keyword) en el ensemble
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")


PROMPT_TEMPLATE = """
Eres un asistente universitario de EAFIT. Responde únicamente basándote
en el contexto proporcionado. Si no encuentras la respuesta en el contexto,
di claramente que no tienes esa información en los documentos disponibles.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""

prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


def load_chunks():
    """Carga y trocea data/ con la misma lógica que ingest.py (para BM25)."""
    docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".md"):
            docs.extend(TextLoader(os.path.join(DATA_DIR, filename), encoding="utf-8").load())

    titles = {}
    for doc in docs:
        title = "Universidad EAFIT"
        for line in doc.page_content.splitlines():
            line = line.strip()
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break
        titles[doc.metadata.get("source", "")] = title

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        title = titles.get(source, "Universidad EAFIT")
        # Slug del path: convierte el filename en palabras buscables por BM25
        slug = os.path.splitext(os.path.basename(source))[0]
        slug_words = re.sub(r"[_/\-]+", " ", slug.replace("www eafit edu co  ", ""))
        chunk.page_content = f"[{title}] {slug_words}\n{chunk.page_content}"
    return chunks


LISTING_FILES = {"www_eafit_edu_co__pregrados.md"}


def is_listing_file(source: str) -> bool:
    name = os.path.basename(source)
    return name.startswith("_indice_") or name in LISTING_FILES


def expand_listing_files(docs):
    """Si un chunk pertenece a un archivo lista/índice, reemplaza por el archivo completo."""
    seen_listings = set()
    out = []
    for d in docs:
        source = d.metadata.get("source", "")
        if is_listing_file(source):
            if source not in seen_listings:
                seen_listings.add(source)
                full = open(source, encoding="utf-8").read()
                name = os.path.basename(source).removesuffix(".md").replace("_", " ")
                out.append(Document(page_content=f"[{name}]\n{full}", metadata={"source": source}))
        else:
            out.append(d)
    return out


def bm25_preprocess(text: str):
    """Tokeniza separando URLs, guiones y barras, lowercase."""
    return re.findall(r"[a-záéíóúñü0-9]+", text.lower())


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
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY)

    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    print(" Indexando BM25 (keyword)...")
    chunks = load_chunks()
    bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=bm25_preprocess)
    bm25_retriever.k = TOP_K
    print(f" BM25 listo ({len(chunks)} chunks)")

    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[BM25_WEIGHT, 1 - BM25_WEIGHT],
    )
    expanded = retriever | RunnableLambda(expand_listing_files)

    chain = (
        {"context": expanded | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print(" Cadena RAG híbrida lista (BM25 + vector)")
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
        print(f"  [{i}] {source}")
        sources.append(source)
    return response, sources


if __name__ == "__main__":
    print("═" * 50)
    print("  Retriever — RAG EAFIT (hybrid)")
    print("═" * 50 + "\n")

    if not GOOGLE_API_KEY:
        raise ValueError("Falta GOOGLE_API_KEY. Agrégala al archivo .env")

    vectorstore      = load_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    query(chain, retriever, "¿Qué pregrados ofrece EAFIT?")
