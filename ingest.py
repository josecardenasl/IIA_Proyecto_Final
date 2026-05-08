import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


PDF_DIR      = "data/pdfs"
VECTORSTORE  = "vectorstore/faiss_index"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE   = 500
CHUNK_OVERLAP = 80


def load_pdfs(directory: str):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            path = os.path.join(directory, filename)
            loader = PyMuPDFLoader(path)
            docs.extend(loader.load())
            print(f"  ✓ Cargado: {filename} ({len(docs)} páginas acumuladas)")
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
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE)
    print(f"  ✓ Vector store guardado en '{VECTORSTORE}'")
    return vectorstore


if __name__ == "__main__":
    print("═" * 50)
    print("  Ingesta de PDFs — RAG EAFIT")
    print("═" * 50)

    print("\n[1/3] Cargando PDFs...")
    docs = load_pdfs(PDF_DIR)

    print("\n[2/3] Dividiendo en chunks...")
    chunks = split_documents(docs)

    print("\n[3/3] Construyendo vector store...")
    build_vectorstore(chunks)

    print("\n Ingesta completada. Ya puedes correr el retriever.")
