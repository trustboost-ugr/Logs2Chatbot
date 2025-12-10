# generate_rasa_files_gemini.py
"""
Batch‑send Rasa‑generation prompts to **Gemini 1.5 Pro** via LangChain’s
`ChatGoogleGenerativeAI`, skipping completed chunks and pacing requests to
avoid Google rate limits (60 requests/min or 32 000 tokens/min at time of
writing).

Usage example
-------------
```bash
export GOOGLE_API_KEY="AIza…"       # how to obtain: see docs below
python generate_rasa_files_gemini.py \
    --prompts_dir output/gpt4o_new \
    --pattern "chunk_*_prompt.txt" \
    --output_dir output/gemini15_new \
    --model gemini-1.5-pro-latest
```

Key features
============
* **Skip** chunks that already have both `.txt` and `.json` responses.
* **Sleep** `--sleep` seconds after each successful call (default 2 s → ~30 req/min).
* **Auto‑retry** on 429/quota errors with exponential back‑off.
* **max_tokens** maps to `max_output_tokens`; `0` = leave unset (Gemini default 8192).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import BaseMessage

# ---------------------------------------------------------------------------

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _tok_count(s: str) -> int:
        return len(_enc.encode(s))
except Exception:
    def _tok_count(s: str) -> int:
        return len(s.split())

REQUIRED_STRINGS = [
    "nlu.yml", "domain.yml", "stories.yml",
    "Please process the following dialogues",
    #"À partir de ces dialogues, générez"
]

SOFT_MAX_INPUT_TOKENS = 150_000  # adjust if you’re confident in your quota

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_prompt_files(prompts_dir: Path, glob_pattern: str) -> Iterator[Path]:
    """Yield prompt files ordered by filename (chunk index)."""
    yield from sorted(prompts_dir.glob(glob_pattern))


def run_gemini(
    prompt: str,
    model: str = "gemini-1.5-pro-latest",
    temperature: float = 0.2,
    max_tokens: int | None = 8192,
    max_retries: int = 5,
    backoff: float = 5.0,
) -> Tuple[str, BaseMessage]:
    """Send a prompt to Gemini with simple back‑off handling."""

    llm_kwargs = {
        "model": model,
        "temperature": temperature,
    }
    if max_tokens is not None and max_tokens > 0:
        llm_kwargs["max_output_tokens"] = max_tokens

    llm = ChatGoogleGenerativeAI(**llm_kwargs)

    attempt = 0
    while True:
        try:
            response: BaseMessage = llm.invoke(prompt)
            return response.content, response
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if ("429" in msg or "rate" in msg or "quota" in msg) and attempt < max_retries:
                wait = backoff * (2 ** attempt)
                print(f"        [RATE-LIMIT] sleeping {wait:.1f}s then retry…")
                time.sleep(wait)
                attempt += 1
                continue
            raise

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send Rasa-generation prompts through Gemini 1.5 Pro",
    )
    parser.add_argument("--prompts_dir", required=True, help="Dir with *_prompt.txt files")
    parser.add_argument("--pattern", default="chunk_*_prompt.txt", help="Glob pattern for prompt files")
    parser.add_argument("--output_dir", required=True, help="Where to write responses")
    parser.add_argument("--model", default="gemini-1.5-pro-latest", help="Gemini model id")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=8192, help="max_output_tokens; 0 to leave unset")
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to wait after each successful request")
    parser.add_argument("--dry_run", action="store_true", help="Skip API calls (debug)")

    args = parser.parse_args()

    prompts_dir = Path(args.prompts_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for prompt_file in iter_prompt_files(prompts_dir, args.pattern):
        chunk_id = prompt_file.stem.replace("_prompt", "")
        print(f"[RUN] {chunk_id} → Gemini")

        out_txt = out_dir / f"{chunk_id}_response.txt"
        out_json = out_dir / f"{chunk_id}_response.json"
        if out_txt.exists() and out_json.exists():
            print("    [SKIP] response already exists – skipping.")
            continue

        if args.dry_run:
            print(f"    [DRY-RUN] prompt size {prompt_file.stat().st_size} bytes")
            continue

        prompt_text = prompt_file.read_text(encoding="utf-8")

        # Guard 1: header sanity
        missing = [s for s in REQUIRED_STRINGS if s not in prompt_text]
        if missing:
            print(
                f"    [SKIP] Prompt missing required strings {missing} → looks like a dialogues-only split. Not sending.")
            continue

        # Guard 2: rough token count and soft cap
        est_tokens = _tok_count(prompt_text)
        print(f"    [INFO] estimated input tokens: {est_tokens:,}")
        if est_tokens > SOFT_MAX_INPUT_TOKENS:
            print(f"    [SKIP] Over soft cap ({SOFT_MAX_INPUT_TOKENS:,}). Rechunk or use the *small* prompts folder.")
            continue

        try:
            content, raw_msg = run_gemini(
                prompt=prompt_text,
                model=args.model,
                temperature=args.temperature,
                max_tokens=(None if args.max_tokens == 0 else args.max_tokens),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    [ERROR] Gemini call failed: {exc}")
            continue

        out_txt.write_text(content, encoding="utf-8")
        with out_json.open("w", encoding="utf-8") as fp:
            json.dump(raw_msg.model_dump(), fp, ensure_ascii=False, indent=2)
        print(f"    [WRITE] Saved to {out_txt.relative_to(out_dir.parent)} & .json")

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("All done.")

# ---------------------------------------------------------------------------
# Notes on obtaining a Gemini API key
# ---------------------------------------------------------------------------
"""
How to obtain GOOGLE_API_KEY
---------------------------
1. Visit **https://makersuite.google.com/app/apikey** (or
   **https://ai.google.dev** ➜ *Get API key*).
2. Sign in with your Google account.
3. Press **Create API key** > copy the string starting with `AIza`. Save it
   securely – you won’t see it again.
4. Export in your shell before running this script:

   ```bash
   export GOOGLE_API_KEY="AIza…"
   ```

The key is per Google Cloud project; usage is currently free within generous
quotas but subject to change.
"""

if __name__ == "__main__":
    main()
