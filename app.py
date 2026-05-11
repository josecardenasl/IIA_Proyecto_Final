import gradio as gr

# Cambiar el import para cambiar el modelo:
from retriever_gemini import load_vectorstore, build_rag_chain, query  # Gemini (requiere GOOGLE_API_KEY en .env)
# from retriever import load_vectorstore, build_rag_chain, query        # Ollama (local, requiere ollama corriendo)

COLLECTION_NAME = "eafit_docs"

vectorstore      = load_vectorstore()
chain, retriever = build_rag_chain(vectorstore)


def chat(question: str, history: list, _context: str):
    response, sources = query(chain, retriever, question)

    if sources:
        unique_sources = list(dict.fromkeys(sources))
        sources_md = "  ".join(f"`{s}`" for s in unique_sources)
        return f"{response}\n\n---\n📄 **Fuentes:** {sources_md}"

    return response


demo = gr.ChatInterface(
    fn=chat,
    title="Asistente Universitario EAFIT",
    description="El asistente responde **únicamente** con información de la documentación oficial de EAFIT.",
    additional_inputs=[
        gr.Textbox(value=COLLECTION_NAME, visible=False)
    ],
    examples=[
        ["¿Qué programas de pregrado ofrece EAFIT?"],
        ["¿Cómo puedo aplicar a una beca en EAFIT?"],
        ["¿Qué servicios ofrece bienestar universitario?"],
        ["¿Cuáles son los requisitos de admisión para pregrado?"],
        ["¿Qué apoyos existen para salud mental en EAFIT?"],
        ["¿Qué becas por mérito académico ofrece EAFIT?"],
        ["¿Qué es EAFIT y cuál es su misión?"],
    ],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
