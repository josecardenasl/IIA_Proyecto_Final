"""Baja todas las URLs de scrapping/urls.txt vía Jina y guarda en data/."""
import asyncio
import aiohttp
import re
from pathlib import Path
from urllib.parse import urlparse

JINA_BASE = "https://r.jina.ai/"
URLS_FILE = Path("scrapping/urls.txt")
OUTPUT_DIR = Path("data")
CONCURRENCY = 3
DELAY = 1.5
MAX_RETRIES = 3


def url_to_filename(url: str) -> str:
    if "wikipedia.org" in url:
        slug = url.rstrip("/").split("/")[-1].lower()
        return f"wikipedia__{slug}.md"
    parsed = urlparse(url)
    path = parsed.netloc + parsed.path
    path = re.sub(r"[^\w\-/]", "_", path).strip("/").replace("/", "__")
    return (path or "index") + ".md"


async def fetch(session, url, sem, idx, total):
    filename = url_to_filename(url)
    filepath = OUTPUT_DIR / filename
    if filepath.exists():
        print(f"  [{idx}/{total}] [skip] {filename}")
        return

    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                await asyncio.sleep(DELAY)
                async with session.get(JINA_BASE + url,
                                       headers={"Accept": "text/markdown"},
                                       timeout=aiohttp.ClientTimeout(total=45)) as r:
                    text = await r.text()
                    if r.status == 200 and len(text) > 500:
                        filepath.write_text(text, encoding="utf-8")
                        print(f"  [{idx}/{total}] [{r.status}] {filename} — {len(text):,} chars")
                        return
                    elif r.status == 429:
                        wait = 2 ** (attempt + 3)
                        print(f"  [{idx}/{total}] [429] {url} — esperando {wait}s")
                        await asyncio.sleep(wait)
                    else:
                        print(f"  [{idx}/{total}] [!] {r.status} {url}")
                        return
            except Exception as e:
                print(f"  [{idx}/{total}] [!] {url}: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)


async def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    urls = [u.strip() for u in URLS_FILE.read_text().splitlines() if u.strip()]
    print(f"Bajando {len(urls)} URLs (concurrency={CONCURRENCY})...\n")

    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, u, sem, i+1, len(urls)) for i, u in enumerate(urls)]
        await asyncio.gather(*tasks)

    saved = len(list(OUTPUT_DIR.glob("*.md")))
    print(f"\nListo. {saved} archivos en {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
