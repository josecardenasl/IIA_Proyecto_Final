import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

DATA_DIR        = "data/"
VECTORSTORE     = "vectorstore/qdrant"
EMBED_MODEL     = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "eafit_docs"
EMBED_DIM       = 384
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 80


def load_data(directory: str):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".md"):
            path = os.path.join(directory, filename)
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
            print(f"   Cargado: {filename} ({len(docs)} docs acumulados)")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"\n  → {len(docs)} páginas divididas en {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks):
    print("\n  Generando embeddings (puede tardar unos minutos)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )

    # Cliente local — guarda en disco, sin servidor
    client = QdrantClient(path=VECTORSTORE)

    # Crea la colección si no existe
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBED_DIM,
            distance=Distance.COSINE
        )
    )

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    vectorstore.add_documents(chunks)
    print(f"   Vector store guardado en '{VECTORSTORE}'")
    return vectorstore


if __name__ == "__main__":
    print("═" * 50)
    print("  Ingesta de PDFs — RAG EAFIT")
    print("═" * 50)

    print("\n[1/3] Cargando datos...")
    docs = load_data(DATA_DIR)

    print("\n[2/3] Dividiendo en chunks...")
    chunks = split_documents(docs)

    print("\n[3/3] Construyendo vector store...")
    build_vectorstore(chunks)

    print("\n Ingesta completada. Ya puedes correr el retriever.")
