"""Load raw handbook markdown files into LangChain Documents."""

import re
from pathlib import Path

import yaml
from langchain_core.documents import Document

# Hugo shortcodes ({{< ... >}} / {{% ... %}}) are noise for retrieval.
_SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)
_FRONT_MATTER =re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)


def split_front_matter(raw: str) -> tuple[dict, str]:
    """Return (front_matter_dict, body). Malformed/absent front matter -> ({}, raw)."""
    m = _FRONT_MATTER.match(raw)
    if not m:
        return {}, raw
    try:
        meta = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return {}, raw[m.end():]
    return (meta if isinstance(meta, dict) else {}), raw[m.end():]


def load_documents(root: Path) -> list[Document]:
    """Walk `root` for .md files; return Documents with `source` metadata."""
    root = Path(root)
    docs: list[Document] = []
    for path in sorted(root.rglob("*.md")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        meta, body = split_front_matter(raw)
        body = _SHORTCODE.sub("", body)
        if not body.strip():
            continue
        title = meta.get("title")
        docs.append(
            Document(
                page_content=body,
                metadata={
                    "source": path.relative_to(root).as_posix(),
                    "title": str(title).strip() if title else "",
                },
            )
        )
    return docs
