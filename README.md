# Asistente RAG para Universidad EAFIT

Sistema de pregunta-respuesta sobre información oficial de EAFIT usando Retrieval-Augmented Generation (RAG) con LangChain, Qdrant y Gemini 2.5 Flash (o Ollama llama3.2 como alternativa local).

---

## 1. Planteamiento y objetivo

### Problema

La Universidad EAFIT publica toda su información oficial (carreras, becas, reglamentos, bienestar, admisiones) en su sitio web `www.eafit.edu.co`, distribuida en **cientos de páginas** organizadas jerárquicamente. Los aspirantes, estudiantes y empleados que buscan información concreta —*¿cuánto cuesta Ingeniería de Sistemas?*, *¿qué pasa si pierdo una asignatura?*, *¿qué becas existen?*— tienen que:

- Navegar manualmente por menús anidados
- Leer páginas completas para extraer un dato puntual
- Consultar reglamentos extensos en PDF/HTML
- Hacer llamadas o esperar respuestas por correo

Esto es ineficiente tanto para el usuario como para la institución (saturación de canales de atención).

### Objetivo general

Construir un **asistente conversacional** que responda preguntas en lenguaje natural sobre información oficial de EAFIT, fundamentando cada respuesta en los documentos reales del sitio, sin alucinar y citando las fuentes.

### Objetivos específicos

1. **Recolectar** un corpus completo y limpio de información institucional desde el sitio oficial.
2. **Indexar** ese corpus en una base de datos vectorial que permita recuperación semántica eficiente.
3. **Implementar** un pipeline RAG híbrido (búsqueda semántica + por palabras clave) que recupere los documentos más relevantes para cada pregunta.
4. **Generar** respuestas precisas usando un LLM (Gemini 2.5 Flash o Ollama llama3.2) con el contexto recuperado.
5. **Exponer** la solución mediante una interfaz web simple (Gradio) que muestre las fuentes utilizadas.

### Alcance

✅ Información estática institucional: pregrados, posgrados, becas, reglamentos, bienestar, internacionalización.
❌ Información transaccional en tiempo real (notas, matrículas, calendarios académicos vigentes) — fuera del alcance de un RAG estático.

---

## 2. Metodología y diagrama de flujo

### Enfoque general

Pipeline **clásico de RAG** dividido en tres etapas independientes que se ejecutan en momentos distintos:

| Etapa | Cuándo se ejecuta | Output |
|---|---|---|
| **Scraping + limpieza** | Una vez (o cuando se actualiza el corpus) | `data/*.md` |
| **Ingest (indexación)** | Una vez tras scraping/cambios | Vector store en Qdrant |
| **Retrieval + generación** | Cada pregunta del usuario | Respuesta + fuentes |

### Diagrama de flujo completo

```
┌─────────────────────────────────────────────────────────────────────┐
│                       FASE 1 — SCRAPING (offline)                   │
│                                                                     │
│  sitemap.xml (4.264 URLs)                                           │
│         │                                                           │
│         ▼                                                           │
│  sitemap_filter.py ──► urls.txt (695 URLs útiles)                   │
│         │                                                           │
│         ▼                                                           │
│  fetch_from_list.py + fetch_pregrados.py (via Jina)                 │
│         │                                                           │
│         ▼                                                           │
│  data/*.md (638 archivos crudos)                                    │
│         │                                                           │
│         ▼                                                           │
│  clean.py (elimina nav, footer, accesibilidad)                      │
│         │                                                           │
│         ▼                                                           │
│  data/*.md limpios                                                  │
│         │                                                           │
│         ▼                                                           │
│  build_indices.py ──► data/_indice_*.md (9 catálogos)               │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FASE 2 — INGEST (offline)                     │
│                                                                     │
│  data/*.md (638 documentos)                                         │
│         │                                                           │
│         ▼                                                           │
│  TextLoader (carga uno por uno)                                     │
│         │                                                           │
│         ▼                                                           │
│  Contextual chunking:                                               │
│    - Extrae H1 de cada doc                                          │
│    - RecursiveCharacterTextSplitter (1500 / 200)                    │
│    - Prefija cada chunk con [Título]                                │
│         │                                                           │
│         ▼                                                           │
│  5.329 chunks                                                       │
│         │                                                           │
│         ▼                                                           │
│  Embeddings: paraphrase-multilingual-MiniLM-L12-v2 (384 dim)        │
│         │                                                           │
│         ▼                                                           │
│  Qdrant (vectorstore/qdrant/, persistente en disco)                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                FASE 3 — RETRIEVAL + GENERATION (online)             │
│                                                                     │
│  Usuario escribe pregunta en Gradio                                 │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┬──────────────┐                                     │
│  ▼             ▼              ▼                                     │
│  BM25       Vector            (en paralelo)                         │
│  (keyword)  (semántico)                                             │
│  top 12     top 12                                                  │
│  └─────────────┬──────────────┘                                     │
│                ▼                                                    │
│  EnsembleRetriever (50/50, Reciprocal Rank Fusion)                  │
│                │                                                    │
│                ▼                                                    │
│  expand_listing_files                                               │
│  (si hay chunk de _indice_*.md o pregrados.md → trae completo)      │
│                │                                                    │
│                ▼                                                    │
│  Prompt + contexto → LLM (Gemini 2.5 Flash / Ollama llama3.2)       │
│                │                                                    │
│                ▼                                                    │
│  Respuesta + lista de fuentes                                       │
│                │                                                    │
│                ▼                                                    │
│  Gradio muestra al usuario                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Decisiones metodológicas clave

| Decisión | Justificación |
|---|---|
| Scraping vía sitemap (no BFS) | Cobertura completa y dirigida; evita basura como noticias |
| Jina como proxy | Bypassa WAF de Incapsula, simplifica scraping de SPAs |
| Embeddings multilingües | El corpus es 100% español; modelos en inglés generan embeddings malos |
| Hybrid BM25 + vector | Cubre tanto queries semánticas como con keywords distintivos (pregrado vs posgrado) |
| Chunks de 1500 chars | Sweet spot entre granularidad y completitud de la información |
| Expansión de archivos índice | Para queries de tipo "lista todos los X" el RAG debe ver el catálogo completo |
| LLM con prompt restrictivo | "Responde solo basándote en el contexto" minimiza alucinaciones |
| LLM intercambiable | Gemini (cloud) para calidad; Ollama (local) para zero-cost / offline |

---

## 3. Scraping

### Objetivo

Construir un corpus de documentos en markdown a partir del sitio oficial `www.eafit.edu.co` que pueda alimentar el sistema RAG con información sobre pregrados, posgrados, becas, reglamentos, bienestar y todo lo institucional.

### Approach final: scraping dirigido por sitemap

```
sitemap.xml → filtrar URLs útiles → bajar vía Jina → limpiar ruido
```

| Script | Función |
|---|---|
| `sitemap_filter.py` | Lee `eafit.edu.co/sitemap.xml` (4.264 URLs), filtra por patrones útiles y descarta basura → genera `urls.txt` con ~695 URLs |
| `fetch_from_list.py` | Baja las 695 URLs en paralelo (concurrencia=3) usando `r.jina.ai` como proxy de scraping |
| `fetch_pregrados.py` | Baja específicamente las 5 páginas paginadas del listado de pregrados y las combina en un solo archivo |
| `clean.py` | Elimina ruido residual (nav, footer, accesibilidad, links sueltos) de todos los `.md` |

### Problemas enfrentados

#### 🔴 Problema 1: BFS ciego desde la home

El scraper original recorría el sitio en breadth-first desde la home con MAX_DEPTH=4. Resultado: 100+ páginas descargadas pero muchas irrelevantes (noticias viejas, eventos, perfiles de profesores), y se perdían páginas críticas que estaban profundas en el árbol del sitio.

**Solución:** descubrir URLs vía `sitemap.xml`. Los sitios serios mantienen un sitemap para Google con todas las URLs canónicas. Es público, exhaustivo y ya filtra duplicados.

#### 🔴 Problema 2: Mucho ruido en cada página

Cada `.md` traía el menú de navegación, banners de cookies, widget de accesibilidad ("Activar contraste, Tamaño de letra, etc.") y el footer (líneas como "Línea nacional 01 8000...", "Vigilada Mineducación"). El contenido útil quedaba ahogado.

**Solución dual:**
- **Jina con headers `X-Target-Selector` y `X-Remove-Selector`** para que el proxy ya devuelva solo el contenido principal.
- **`clean.py`**: post-procesamiento con regex que elimina líneas que son solo imágenes, solo links, bullets con links, y patrones de accesibilidad. Redujo el tamaño promedio de los archivos en 60-85%.

#### 🔴 Problema 3: Paginación JavaScript (los 26 pregrados)

La página `/pregrados` mostraba solo 5-6 carreras a la vez con un paginador cliente. Jina solo veía la primera página (5 pregrados de 26).

**Solución:** descubrir el patrón de URL paginada (`/pregrados?page=1`, `?page=2`...), iterar las 5 páginas y combinarlas en un solo archivo. Eso es lo que hace `fetch_pregrados.py`.

#### 🔴 Problema 4: Incapsula/Imperva (WAF)

Intenté usar Playwright headless para JavaScript-heavy pages. Incapsula bloqueó casi todo (devolvía error de 84 chars). Jina pasa el WAF porque opera desde un proxy con buen TLS fingerprint.

**Solución:** abandonar Playwright, mantenerse en Jina.

---

## 4. Ingest

### Objetivo

Convertir los 638 archivos markdown en chunks vectorizados almacenados en Qdrant, listos para búsqueda semántica.

### Pipeline

```
data/*.md → load → contextual chunking → embeddings → Qdrant
```

### Cambio 1: Modelo de embeddings

#### 🔴 Antes: `sentence-transformers/all-MiniLM-L6-v2`

Modelo de 384 dimensiones, entrenado solo en **inglés**. Los embeddings de texto en español eran de mala calidad: el modelo no distinguía bien conceptos similares, mezclaba "estudiante" con "profesor", etc. Las respuestas del LLM eran erráticas porque el retriever traía documentos irrelevantes.

#### ✅ Ahora: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

Mismo tamaño (384 dim), pero **entrenado en 50+ idiomas incluyendo español**. Los embeddings capturan mejor el significado en español. La elección de mantener 384 dimensiones permite reutilizar la colección Qdrant sin recrearla.

### Cambio 2: Contextual chunking

#### Problema

Un chunk típico del archivo `pregrados.md` puede ser:
```
9 semestres
$16.060.617
SNIES 1247
Presencial
```

**Sin contexto sobre qué carrera es.** El embedding de ese fragmento es genérico y no le sirve al retriever.

#### Solución

Antes de chunkear, se extrae el **H1 del documento** (`# Pregrados - Universidad EAFIT`) y se prefija a cada chunk:

```python
def split_documents(docs):
    titles = {doc.metadata["source"]: get_doc_title(doc) for doc in docs}
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        title = titles.get(chunk.metadata["source"], "Universidad EAFIT")
        chunk.page_content = f"[{title}]\n{chunk.page_content}"
    return chunks
```

Resultado: el chunk ahora dice:
```
[Pregrados - Universidad EAFIT]
9 semestres
$16.060.617
SNIES 1247
Presencial
```

**Beneficios:**
- El embedding captura que es información sobre pregrados.
- BM25 indexa el título como tokens buscables.
- El LLM recibe contexto explícito de qué archivo viene cada fragmento.

### Cambio 3: Tamaño de chunks

| Parámetro | Antes | Después |
|---|---|---|
| `CHUNK_SIZE` | 500 chars | **1500** chars |
| `CHUNK_OVERLAP` | 80 chars | **200** chars |

**Por qué:** chunks de 500 chars eran demasiado pequeños — una sola carrera no cabía completa en un chunk, mucho menos varias. Con 1500 chars cada chunk de `pregrados.md` cubre 4-5 carreras con sus datos completos. El overlap del 13% asegura que frases que caen en el borde aparezcan completas en al menos un chunk.

---

## 5. Retriever

### Estado inicial

Inicialmente el retriever era **solo búsqueda vectorial** (similarity search en Qdrant), TOP_K=4.

### Síntomas raros

- **"¿Qué es EAFIT?"** → "No tengo esa información." (¡pero EAFIT está en cada documento!)
- **"¿Qué pregrados ofrece EAFIT?"** → devolvía documentos de **posgrados** y maestrías.
- **"¿Qué pasa si pierdo una materia?"** → devolvía talleres artísticos.

### Diagnóstico: los embeddings no discriminan bien en español

1. `pregrado` y `posgrado` son **una sola letra de diferencia** — los embeddings los tratan casi como sinónimos.
2. El usuario dice `materia`, el reglamento dice `asignatura` — solo el vector entiende sinónimos, pero los confunde con conceptos generales como "talleres".
3. El TOP_K bajo (4) no daba margen para corregir errores.

### Solución 1: Hybrid Retrieval (BM25 + Vector)

#### Qué es BM25

Algoritmo de búsqueda **por palabras clave exactas** (similar a SQL LIKE pero con ranking). Calcula un score por documento basado en:

- **TF** (term frequency): cuántas veces aparece la palabra
- **IDF** (inverse document frequency): qué tan rara es la palabra en el corpus
- **Length normalization**: penaliza documentos muy largos

Palabras comunes como "EAFIT" o "el" pesan poco (aparecen en todas partes). Palabras específicas como "pregrado" pesan mucho.

#### Combinación

```python
ensemble = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
```

| Tipo de query | Quién gana |
|---|---|
| "qué es EAFIT" | Vector — entiende intent |
| "¿qué pregrados ofrece?" | BM25 — caza "pregrado" literal |
| "duración ingeniería sistemas" | Ambos colaboran |

**El ensemble suma las fortalezas:** keyword garantiza precisión léxica, vector garantiza comprensión semántica.

#### Detalle técnico: tokenizer custom

El tokenizer default de BM25 no separa bien las URLs. Para `pregrados/escuela-ingenieria` lo veía como un solo token gigante. Fix:

```python
def bm25_preprocess(text: str):
    return re.findall(r"[a-záéíóúñü0-9]+", text.lower())
```

Tokeniza por cualquier cosa que no sea letra/número, lowercase. Así las URLs se descomponen en palabras buscables.

### Solución 2: Aumentar TOP_K de 4 a 12

Más candidatos = más probabilidad de incluir el documento correcto. El LLM filtra el ruido en el contexto.

### Solución 3: Expansión de archivos índice

#### El problema persistente

Aun con hybrid retrieval, la pregunta **"¿Qué pregrados ofrece EAFIT?"** fallaba parcialmente:
- El retriever traía `pregrados.md` en el top
- Pero solo 1 chunk del archivo (los primeros 1500 chars)
- Ese chunk tenía 4-5 carreras, no las 27 completas

#### La solución

Cuando el retriever trae cualquier chunk de un archivo "lista" (`pregrados.md` o cualquier `_indice_*.md`), **se reemplaza ese chunk por el contenido completo del archivo** antes de mandarlo al LLM.

```python
def expand_listing_files(docs):
    seen = set()
    out = []
    for d in docs:
        source = d.metadata.get("source", "")
        if is_listing_file(source):
            if source not in seen:
                seen.add(source)
                full = open(source).read()
                out.append(Document(page_content=full, metadata={"source": source}))
        else:
            out.append(d)
    return out
```

#### Por qué funciona

Los archivos índice son catálogos completos. Cuando el usuario pide "qué pregrados ofrece", el LLM necesita ver **todos los items**, no una muestra. El retriever encuentra el archivo correcto (BM25 + vector), y el expansor garantiza que llegue completo.

Es una **mini Parent Document Retriever** hardcoded para archivos de catálogo.

### Solución 4: Generador de índices (`build_indices.py`)

Para que la solución 3 funcione, **hace falta que existan archivos índice**. Pero el corpus original no los tenía — solo había 28 archivos individuales por carrera, ninguno enumeraba todos.

`build_indices.py` recorre `data/`, agrupa archivos por categoría (prefijo del nombre) y genera 9 archivos `_indice_*.md` con la lista completa:

- `_indice_pregrados.md` (23 entradas)
- `_indice_posgrados.md` (100 entradas)
- `_indice_plan-de-estudio.md` (131 entradas)
- `_indice_bienestar-universitario.md` (131 entradas)
- `_indice_becas-y-financiacion.md` (19 entradas)
- `_indice_institucional.md` (130 entradas)
- `_indice_escuela.md` (36 entradas)
- `_indice_idiomas.md` (33 entradas)
- `_indice_internacionalizacion.md` (15 entradas)

Cada índice contiene `- **Título** — URL` por cada documento de la categoría. Es la "tabla de contenidos" automática del corpus.

### Pipeline final del retriever

```
Pregunta del usuario
       │
       ├──────────┬───────────┐
       ▼          ▼
     BM25      Vector
   (keyword)  (semántico)
   top 12       top 12
       │          │
       └──────┬───┘
              ▼
       Ensemble (50/50)
              ▼
      Expand listing files
              ▼
        LLM (Gemini)
              ▼
         Respuesta
```

---

## 6. Resultados

### Antes de las mejoras

> "¿Qué pregrados ofrece EAFIT?"
> *"No tengo esa información en los documentos disponibles."*

> "¿Qué es EAFIT?"
> *"No tengo esa información..."*

### Después de las mejoras

> "¿Qué pregrados ofrece EAFIT?"
> *"EAFIT ofrece 26 programas de pregrado, entre los cuales se encuentran: Ingeniería de Construcción, Ingeniería Industrial, Psicología, Biología, Ingeniería de Sistemas, ... [los 27 listados completos]"*

> "¿Qué deberes tiene un estudiante?"
> *"Según el Reglamento académico de los programas de pregrado: a) La asistencia a clase... b) Consultar, acatar y respetar la filosofía... c) ... t) Los demás deberes contemplados en este reglamento."* [lista completa de 20 deberes]

---

## 7. Comparación con el chatbot oficial de EAFIT

Como benchmark cualitativo se evaluó el **chatbot actual disponible en `eafit.edu.co`** con las mismas preguntas. Los resultados muestran que nuestro asistente RAG supera al oficial en precisión y completitud.

### Caso de prueba: "¿Qué pregrados tiene EAFIT?"

| Sistema | Respuesta | Correcto |
|---|---|---|
| Chatbot oficial EAFIT | "EAFIT ofrece **15 pregrados**: [lista incompleta]" | ❌ **Falso** — EAFIT tiene 26 programas de pregrado |
| Nuestro RAG | "EAFIT ofrece **26 programas de pregrado**: Ingeniería de Construcción, Ingeniería Industrial, Psicología, Biología, Ingeniería de Sistemas, Ingeniería Agronómica, Ingeniería Física, Geología, Ingeniería Matemática, Música, Ingeniería Civil, Finanzas, Diseño Urbano y Gestión del Hábitat, Ciencias Políticas, Comunicación Social, Economía, Ingeniería Mecánica, Diseño Interactivo, Literatura, Ingeniería de Diseño de Producto, Ingeniería de Producción, Ingeniería de Procesos, Mercadeo, Administración de Negocios, Negocios Internacionales, Contaduría Pública, Derecho" | ✅ Lista completa |

### Por qué nuestro sistema acertó

Es **exactamente el caso que motivó la solución 3 del retriever** (expansión de archivos índice):
- El RAG estándar habría devuelto solo el primer chunk de `pregrados.md` con 4-5 carreras.
- El chatbot oficial probablemente sufre del mismo problema: retrieval que solo trae un fragmento del catálogo.
- Nuestra solución detecta que `pregrados.md` es un archivo de lista y **garantiza que llegue completo** al LLM, evitando respuestas truncadas.

### Implicación

El problema no es la capacidad del LLM (Gemini/llama3.2 son sobradamente capaces de listar 26 nombres). El problema es **cómo se le entrega el contexto**. Sin el expansor de índices, hasta los chatbots de instituciones grandes fallan en preguntas tipo "enumera todos los X". Es un patrón común y subestimado en sistemas RAG en producción.

---

## 8. Stack técnico

| Componente | Herramienta |
|---|---|
| Scraping | `r.jina.ai` (proxy), `aiohttp`, `playwright` |
| Limpieza | regex (`clean.py`) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector store | Qdrant local (file-based) |
| Búsqueda híbrida | `BM25Retriever` + `EnsembleRetriever` (LangChain) |
| LLM | Gemini 2.5 Flash (cloud) o Ollama llama3.2 (local) |
| Framework RAG | LangChain LCEL |
| UI | Gradio `ChatInterface` |
| Gestor de paquetes | `uv` |

---

## 9. Lecciones aprendidas

1. **Los embeddings multilingües son obligatorios para corpus en español.** Un modelo entrenado solo en inglés destruye la calidad del retrieval aunque la similitud coseno se vea decente.

2. **Hybrid retrieval (BM25 + vector) no es opcional.** Cada uno cubre los huecos del otro. Solo vector falla en queries con keywords distintivos. Solo BM25 falla con sinónimos.

3. **El chunk size importa muchísimo.** Chunks demasiado pequeños fragmentan información relacionada. Chunks demasiado grandes diluyen el ranking de BM25. El sweet spot fue 1500 chars para este corpus.

4. **El RAG no puede "agregar" información dispersa.** Si la respuesta requiere enumerar items distribuidos en 28 archivos, hay que crear un archivo índice. El retriever solo recupera, no infiere globalmente.

5. **La calidad del input determina todo.** Sin limpieza del ruido (nav/footer/accesibilidad), el embedding model y el LLM trabajan sobre basura. Invertir en `clean.py` mejoró las respuestas más que cualquier hiperparámetro.

6. **El sitemap es tu amigo.** Antes de hacer BFS ciego, revisa si el sitio tiene `sitemap.xml`. Te da la lista canónica de URLs sin tener que adivinar la estructura.
