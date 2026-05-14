import os
import re

DATA_DIR = "data/"

_ONLY_LINK    = re.compile(r"^\s*\[.*?\]\(.*?\)\s*$")
_ONLY_IMAGE   = re.compile(r"^\s*!\[.*?\]\(.*?\)\s*$")
_BULLET_LINK  = re.compile(r"^\s*\*\s+\[.*?\]\(.*?\)\s*$")
_HEADING_LINK = re.compile(r"^\s*#{1,6}\s+\[.*?\]\(.*?\)\s*$")

_NOISE_STARTS = (
    "accesibilidad",
    "pasar al contenido",
    "cerrar modal",
    "activar contraste",
    "tamaño de letra",
    "habilitar el audio",
    "información para ti",
    "configurar el idioma",
    "iniciar sesión",
    "markdown content:",
    "url source:",
    "title:",
    "published time:",
)


def clean(text: str) -> str:
    lines = text.splitlines()
    out = []
    for line in lines:
        low = line.strip().lower()
        if not low:
            out.append("")
            continue
        if any(low.startswith(s) for s in _NOISE_STARTS):
            continue
        if _ONLY_IMAGE.match(line) or _ONLY_LINK.match(line):
            continue
        if _BULLET_LINK.match(line) or _HEADING_LINK.match(line):
            continue
        out.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


if __name__ == "__main__":
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".md")]
    print(f"Limpiando {len(files)} archivos en '{DATA_DIR}'...")

    for filename in files:
        path = os.path.join(DATA_DIR, filename)
        original = open(path, encoding="utf-8").read()
        cleaned = clean(original)

        reduction = round((1 - len(cleaned) / max(len(original), 1)) * 100)
        print(f"  {filename[:60]:<60} -{reduction}%")

        with open(path, "w", encoding="utf-8") as f:
            f.write(cleaned)

    print("\nListo.")
