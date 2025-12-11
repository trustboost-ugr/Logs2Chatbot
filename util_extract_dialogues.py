import re

_D_RE = re.compile(r"Dialogues:\s*(.+)", re.DOTALL | re.IGNORECASE)


def extract_dialogues(raw_prompt: str) -> str:
    """Return just the dialogue section from an old chunk_*_prompt.txt."""
    m = _D_RE.search(raw_prompt)
    if not m:
        raise ValueError("No 'Dialogues:' section found.")
    return m.group(1).strip()
