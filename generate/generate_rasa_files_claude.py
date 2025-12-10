# --- PATCH: generate_rasa_files_claude.py ---

from __future__ import annotations
import argparse, json, time
from pathlib import Path
from typing import Iterator, Tuple
from langchain_anthropic import ChatAnthropic
from langchain.schema import BaseMessage

# Optional: better token estimate with tiktoken (OpenAI cl100k_base ≈ Claude’s)
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def estimate_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:
    # Fallback: rough heuristic ~4 chars/token
    def estimate_tokens(text: str) -> int:
        return max(1, int(len(text) / 4))

def iter_prompt_files(prompts_dir: Path, glob_pattern: str) -> Iterator[Path]:
    yield from sorted(prompts_dir.glob(glob_pattern))

def run_claude(
    prompt: str,
    model: str = "claude-3-opus-20240229",
    temperature: float = 0.2,
    max_tokens: int | None = 4096,
    max_retries: int = 5,
    backoff: float = 10.0,
) -> Tuple[str, BaseMessage]:
    llm_kwargs = {"model": model, "temperature": temperature}
    if max_tokens is not None:
        llm_kwargs["max_tokens_to_sample"] = max_tokens
    llm = ChatAnthropic(**llm_kwargs)

    attempt = 0
    while True:
        try:
            response: BaseMessage = llm.invoke(prompt)
            return response.content, response
        except Exception as exc:  # robust 429 handling
            msg = str(exc).lower()
            if ("429" in msg or "rate" in msg or "limit" in msg) and attempt < max_retries:
                wait = backoff * (2 ** attempt)
                print(f"        [RATE-LIMIT] sleeping {wait:.1f}s then retry…")
                time.sleep(wait)
                attempt += 1
                continue
            raise

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Rasa-generation prompts through Claude 3 via LangChain")
    parser.add_argument("--prompts_dir", required=True)
    parser.add_argument("--pattern", default="chunk_*_prompt.txt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="claude-3-opus-20240229")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=4096, help="max_tokens_to_sample; 0 = Claude default")
    # NEW: TPM pacing
    parser.add_argument("--tpm", type=int, default=30000, help="Input tokens per minute (Claude Opus ≈ 30k)")
    parser.add_argument("--safety", type=float, default=1.2, help="Safety multiplier on computed sleep (e.g., 1.2)")
    # (legacy sleep kept as floor)
    parser.add_argument("--sleep", type=float, default=4.0, help="Minimum seconds to wait after each successful request")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    prompts_dir = Path(args.prompts_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for prompt_file in iter_prompt_files(prompts_dir, args.pattern):
        chunk_id = prompt_file.stem.replace("_prompt", "")
        print(f"[RUN] {chunk_id} → Claude")

        out_txt = out_dir / f"{chunk_id}_response.txt"
        out_json = out_dir / f"{chunk_id}_response.json"
        if out_txt.exists() and out_json.exists():
            print("    [SKIP] response already exists – skipping.")
            continue

        prompt_text = prompt_file.read_text(encoding="utf-8")
        tokens_in = estimate_tokens(prompt_text)
        print(f"    [INFO] estimated input tokens: {tokens_in:,}")

        if args.dry_run:
            print("    [DRY-RUN] skipping API call")
            continue

        # Call Claude
        try:
            content, raw_msg = run_claude(
                prompt=prompt_text,
                model=args.model,
                temperature=args.temperature,
                max_tokens=(None if args.max_tokens == 0 else args.max_tokens),
            )
        except Exception as exc:
            print(f"    [ERROR] Claude call failed: {exc}")
            continue

        # Save outputs
        out_txt.write_text(content, encoding="utf-8")
        with out_json.open("w", encoding="utf-8") as fp:
            json.dump(raw_msg.dict(), fp, ensure_ascii=False, indent=2)
        print(f"    [WRITE] Saved {out_txt.name} & {out_json.name}")

        # Dynamic TPM pacing:
        seconds_for_tpm = (tokens_in / float(args.tpm)) * 60.0
        sleep_needed = max(args.sleep, seconds_for_tpm * args.safety)
        print(f"    [PACE] sleeping {sleep_needed:.1f}s to respect ~{args.tpm}/min TPM")
        time.sleep(sleep_needed)

    print("All done.")

if __name__ == "__main__":
    main()
