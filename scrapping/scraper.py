import asyncio
import aiohttp
import re
import os
from urllib.parse import urljoin, urlparse
from pathlib import Path
from collections import deque

SEED_URL = "https://www.eafit.edu.co/"
JINA_BASE = "https://r.jina.ai/"
MAX_DEPTH = 4
MAX_CONCURRENT = 1            # un request a la vez
DELAY_BETWEEN_REQUESTS = 3.0  # segundos entre requests
MAX_RETRIES = 4
OUTPUT_DIR = Path("output")
ALLOWED_DOMAIN = "eafit.edu.co"

# Circuit breaker
CB_THRESHOLD = 3   # 429s consecutivos para abrir el circuito
CB_COOLDOWN = 60   # segundos de pausa global cuando se abre

visited = set()
semaphore = asyncio.Semaphore(MAX_CONCURRENT)

cb_consecutive_failures = 0
cb_open = False


def url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.netloc + parsed.path
    path = re.sub(r'[^\w\-/]', '_', path)
    path = path.strip("/").replace("/", "__")
    return (path or "index") + ".md"


def extract_links(markdown: str, base_url: str) -> list[str]:
    """Extrae links markdown [text](url) y URLs crudas del contenido."""
    links = []
    # Links markdown: [texto](url)
    for url in re.findall(r'\[.*?\]\((https?://[^)]+)\)', markdown):
        links.append(url)
    # URLs crudas en el texto
    for url in re.findall(r'(?<!\()https?://[^\s)\]"\']+', markdown):
        links.append(url)
    
    filtered = []
    for url in links:
        url = url.split('#')[0].rstrip('.,;')
        if not url:
            continue
        parsed = urlparse(url)
        if ALLOWED_DOMAIN in parsed.netloc:
            filtered.append(url)
    
    return list(set(filtered))


async def fetch_page(session: aiohttp.ClientSession, url: str) -> str | None:
    global cb_consecutive_failures, cb_open

    jina_url = JINA_BASE + url
    headers = {"Accept": "text/markdown"}

    async with semaphore:
        # Si el circuito está abierto, esperar el cooldown
        if cb_open:
            print(f"  [CB] Circuito abierto — pausando {CB_COOLDOWN}s antes de reintentar...")
            await asyncio.sleep(CB_COOLDOWN)
            cb_open = False
            cb_consecutive_failures = 0

        for attempt in range(MAX_RETRIES):
            try:
                await asyncio.sleep(DELAY_BETWEEN_REQUESTS)
                async with session.get(jina_url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        cb_consecutive_failures = 0  # reset al tener éxito
                        return await resp.text()
                    elif resp.status == 429:
                        cb_consecutive_failures += 1
                        wait = 2 ** (attempt + 2)  # 4s, 8s, 16s, 32s
                        print(f"  [429] Rate limit en {url} — esperando {wait}s (intento {attempt+1}/{MAX_RETRIES})")

                        if cb_consecutive_failures >= CB_THRESHOLD:
                            print(f"  [CB] {cb_consecutive_failures} fallos consecutivos — abriendo circuito")
                            cb_open = True
                            await asyncio.sleep(CB_COOLDOWN)
                            cb_open = False
                            cb_consecutive_failures = 0
                        else:
                            await asyncio.sleep(wait)
                    else:
                        print(f"  [!] HTTP {resp.status} -> {url}")
                        return None
            except Exception as e:
                print(f"  [!] Error en {url}: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)

        print(f"  [✗] Abandonado tras {MAX_RETRIES} intentos: {url}")
        return None


async def scrape(session: aiohttp.ClientSession, url: str, depth: int):
    if url in visited or depth > MAX_DEPTH:
        return
    visited.add(url)

    indent = "  " * depth
    print(f"{indent}[{depth}] {url}")

    content = await fetch_page(session, url)
    if not content:
        return

    # Guardar archivo
    filename = url_to_filename(url)
    filepath = OUTPUT_DIR / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")
    print(f"{indent}    -> guardado: {filename}")

    if depth < MAX_DEPTH:
        links = extract_links(content, url)
        tasks = [scrape(session, link, depth + 1) for link in links if link not in visited]
        await asyncio.gather(*tasks)


async def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Iniciando scraping desde: {SEED_URL}")
    print(f"Profundidad máxima: {MAX_DEPTH} | Concurrencia: {MAX_CONCURRENT}\n")

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)
    async with aiohttp.ClientSession(connector=connector) as session:
        await scrape(session, SEED_URL, depth=0)

    print(f"\nListo. {len(visited)} páginas visitadas.")
    print(f"Archivos guardados en: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())