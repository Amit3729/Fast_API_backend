from typing import List
import re


def fixed_chunk(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split by characters with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start < 0:
            start = 0
    return [c for c in chunks if c]


def paragraph_chunk(text: str, max_chunk_chars: int = 1500) -> List[str]:
    """Split by paragraphs; join small paragraphs until reaching limit."""
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    buffer = []
    buffer_len = 0
    for p in paras:
        if buffer_len + len(p) <= max_chunk_chars:
            buffer.append(p)
            buffer_len += len(p)
        else:
            if buffer:
                chunks.append("\n\n".join(buffer))
            buffer = [p]
            buffer_len = len(p)
    if buffer:
        chunks.append("\n\n".join(buffer))
    return chunks


def chunk_text(text: str, strategy: str = "fixed") -> List[str]:
    """Strategy selector wrapper."""
    strategy = strategy.lower()

    if strategy in ["fixed", "simple"]:
        return fixed_chunk(text)
    elif strategy == "paragraph":
        return paragraph_chunk(text)
    else:
        raise ValueError("Invalid strategy. Use 'fixed', 'simple', or 'paragraph'.")

