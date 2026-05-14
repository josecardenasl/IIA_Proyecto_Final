"""Descarga todas las páginas paginadas del listado de pregrados y las combina."""
import asyncio
import socket
import aiohttp
import re
from pathlib import Path

JINA_BASE = "https://r.jina.ai/"
OUTPUT    = Path("data/www_eafit_edu_co__pregrados.md")
BASE_URL  = "https://www.eafit.edu.co/pregrados"
PAGES     = 5

NOISE = re.compile(
    r"^\s*("
    r"!\[.*?\]\(.*?\)"
    r"|\[.*?\]\(.*?\)"
    r"|\*\s+\[.*?\]\(.*?\)"
    r"|Accesibilidad.*"
    r"|Pasar al contenido.*"
    r"|Cerrar modal.*"
    r"|Más información Activar.*"
    r"|Mostrar el contenido.*"
    r"|Habilitar el audio.*"
    r"|Configurar el idioma.*"
    r"|Idioma\s*"
    r"|Title:.*"
    r"|URL Source:.*"
    r"|Markdown Content:.*"
    r"|Facet .*"
    r")\s*$"
)


def clean(text: str) -> str:
    lines = [l for l in text.splitlines() if not NOISE.match(l)]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def extract_programs(text: str) -> str:
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if "#### " in l), 0)
    end   = next((i for i, l in enumerate(lines) if "Guía de aspirantes" in l), len(lines))
    return "\n".join(lines[start:end]).strip()


async def fetch_page(session: aiohttp.ClientSession, page: int) -> str:
    target = BASE_URL if page == 0 else f"{BASE_URL}?label=&page={page}"
    url = f"{JINA_BASE}{target}"
    async with session.get(url, headers={"Accept": "text/markdown"},
                           timeout=aiohttp.ClientTimeout(total=45)) as r:
        print(f"  Página {page+1}/{PAGES} — status {r.status}")
        return await r.text()


async def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    connector = aiohttp.TCPConnector(family=socket.AF_INET)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_page(session, p) for p in range(PAGES)]
        pages = await asyncio.gather(*tasks)

    sections = [s for s in (extract_programs(clean(raw)) for raw in pages) if s]
    combined = (
        "# Pregrados - Universidad EAFIT\n\n"
        "EAFIT ofrece 26 programas de pregrado distribuidos en 5 escuelas.\n\n"
        + "\n\n---\n\n".join(sections)
    )

    OUTPUT.write_text(combined, encoding="utf-8")
    print(f"\nGuardado en {OUTPUT} ({len(combined):,} chars)")


if __name__ == "__main__":
    asyncio.run(main())
