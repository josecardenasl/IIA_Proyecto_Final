"""Genera archivos índice por categoría a partir de data/*.md."""
import re
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data")

CATEGORIES = {
    "pregrados":            "Pregrados",
    "posgrados":            "Posgrados",
    "plan-de-estudio":      "Planes de Estudio",
    "bienestar-universitario": "Bienestar Universitario",
    "becas-y-financiacion": "Becas y Financiación",
    "institucional":        "Información Institucional",
    "escuela-":             "Escuelas",
    "idiomas":              "Idiomas",
    "internacionalizacion": "Internacionalización",
}


def get_title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[:30]:
        line = line.strip()
        if line.startswith("# ") and len(line) > 3:
            return line.lstrip("# ").strip()
    return path.stem.replace("www_eafit_edu_co__", "").replace("__", " / ").replace("_", " ")


def url_from_filename(name: str) -> str:
    stem = name.removesuffix(".md")
    if stem.startswith("www_eafit_edu_co__"):
        path = stem.removeprefix("www_eafit_edu_co__").replace("__", "/")
        return f"https://www.eafit.edu.co/{path}"
    return ""


def main():
    groups = defaultdict(list)
    for md in sorted(DATA_DIR.glob("*.md")):
        if md.name.startswith("_indice_"):
            continue
        stem = md.stem.removeprefix("www_eafit_edu_co__")
        for pattern, _ in CATEGORIES.items():
            if stem.startswith(pattern):
                title = get_title(md)
                url = url_from_filename(md.name)
                groups[pattern].append((title, url))
                break

    for pattern, label in CATEGORIES.items():
        items = groups.get(pattern, [])
        if not items:
            continue
        out = DATA_DIR / f"_indice_{pattern.rstrip('-')}.md"
        lines = [f"# Índice de {label} - Universidad EAFIT", ""]
        lines.append(f"EAFIT cuenta con los siguientes {label.lower()}:")
        lines.append("")
        for title, url in items:
            lines.append(f"- **{title}** — {url}" if url else f"- **{title}**")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  {out.name} — {len(items)} entradas")

    print(f"\nÍndices generados en {DATA_DIR.resolve()}")


if __name__ == "__main__":
    main()
