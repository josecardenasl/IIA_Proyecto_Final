"""Baja sitemap de EAFIT, filtra URLs útiles y guarda scrapping/urls.txt."""
import re
import urllib.request
from pathlib import Path

SITEMAP_URL = "https://www.eafit.edu.co/sitemap.xml"
OUTPUT = Path("scrapping/urls.txt")

INCLUDE_PATTERNS = [
    r"/pregrados(/|$|\?)",
    r"/posgrados(/|$|\?)",
    r"/plan-de-estudio",
    r"/bienestar-universitario",
    r"/becas-y-financiacion",
    r"/inscripciones",
    r"/institucional",
    r"/escuela-",
    r"/idiomas",
    r"/internacionalizacion",
    r"/faq",
    r"/nuestras-sedes",
    r"/registro-academico",
]

EXCLUDE_PATTERNS = [
    r"^https://www\.eafit\.edu\.co/en",
    r"/comunicados-institucionales",
    r"/noticias",
    r"/nuestros-profesores",
    r"/calendario-academico",
    r"/informes-de-gestion",
]

EXTRA_URLS = [
    "https://es.wikipedia.org/wiki/Universidad_EAFIT",
]


def main():
    print(f"Descargando sitemap: {SITEMAP_URL}")
    req = urllib.request.Request(SITEMAP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        xml = r.read().decode("utf-8")

    urls = re.findall(r"<loc>([^<]+)</loc>", xml)
    print(f"  {len(urls)} URLs totales en sitemap")

    include_re = re.compile("|".join(INCLUDE_PATTERNS))
    exclude_re = re.compile("|".join(EXCLUDE_PATTERNS))

    filtered = [u for u in urls if include_re.search(u) and not exclude_re.search(u)]
    filtered = sorted(set(filtered + EXTRA_URLS))

    OUTPUT.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    print(f"  {len(filtered)} URLs filtradas → {OUTPUT}")


if __name__ == "__main__":
    main()
