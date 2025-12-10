#!/usr/bin/env python3
"""
generate_rasa_files_llamacpp_seq.py

Sequential pipeline:

    1) Ask llama.cpp for nlu.yml
    2) Feed that nlu.yml back to llama.cpp → domain.yml
    3) Feed nlu.yml + domain.yml back → stories.yml

Each result is saved in   <output_dir>/<chunk_id>/<file>.yml
and the script skips any stage that already exists (safe resume).

It expects:
* template files in   templates_rasa/{nlu,domain,stories}.tmpl
* original chunk_*_prompt.txt files that contain a 'Dialogues:' block
* llama.cpp server exposing the OpenAI-style chat endpoint
  (e.g. http://127.0.0.1:8084/v1/chat/completions)
"""
from __future__ import annotations

import argparse
import json
import textwrap
import time
from pathlib import Path

import requests

from util_extract_dialogues import extract_dialogues  # helper you created

# ---------------------------------------------------------------------------

SYSTEM_MSG = "You are a helpful assistant that outputs valid YAML for Rasa."
# SYSTEM_MSG = (
#   "You are a careful assistant that outputs ONLY valid YAML for Rasa 3.x, "
#   "no prose before/after. Dialogues are in French. "
#   "Use English identifiers for intents/entities/slots (e.g., request_hotel_info, inform_city), "
#   "but example utterances can remain in French. "
#   "Do not include markdown fences, comments, or explanations."
# )


#DEFAULT_MODEL = "mistral-small-3.2-24b-instruct-2506-q5_k_m.gguf"
DEFAULT_MODEL = "Mistral-Small-3.2-24B-Instruct-2506-Q6_K_L.gguf"

def call_llama(base_url: str, model_name: str, messages: list[dict], max_tokens: int, temp: float) -> str:
    """Wrapper for llama.cpp's chat endpoint."""
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_name,
            "temperature": temp,
            "max_tokens": max_tokens,
            "messages": messages,
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# ---------------------------------------------------------------------------

def load_template(tmpl_dir: Path, name: str) -> str:
    return (tmpl_dir / name).read_text(encoding="utf-8")

# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_dir", required=True)
    ap.add_argument("--pattern", default="chunk_*_prompt.txt")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--templates_dir", default="templates_rasa")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--base_url", default="http://127.0.0.1:8084")
    ap.add_argument("--model_name", default=DEFAULT_MODEL)
    ap.add_argument("--max_prompt_chars", type=int, default=12000,
                    help="Hard cap on dialogues text per request (prevents ctx overflow)")

    args = ap.parse_args()

    prompts_dir  = Path(args.prompts_dir)
    out_root     = Path(args.output_dir);   out_root.mkdir(parents=True, exist_ok=True)
    tmpl_dir     = Path(args.templates_dir)

    # preload templates once
    tmpl_nlu     = load_template(tmpl_dir, "nlu.tmpl")
    tmpl_domain  = load_template(tmpl_dir, "domain.tmpl")
    tmpl_stories = load_template(tmpl_dir, "stories.tmpl")

    for prompt_file in sorted(prompts_dir.glob(args.pattern)):
        chunk_id  = prompt_file.stem.replace("_prompt", "")   # "chunk_001" -> "001"
        chunk_dir = out_root / chunk_id
        chunk_dir.mkdir(parents=True, exist_ok=True)

        # --- extract only the dialogues from the old prompt ----------------
        raw_prompt = prompt_file.read_text(encoding="utf-8")
        dialogues  = extract_dialogues(raw_prompt)
        if len(dialogues) > args.max_prompt_chars:
            print(f"[WARN] {chunk_id}: truncating dialogues from {len(dialogues)} to {args.max_prompt_chars} chars")
            dialogues = dialogues[:args.max_prompt_chars]

        # -------------------------------------------------------------------#
        # 1) NLU -------------------------------------------------------------
        # -------------------------------------------------------------------#
        nlu_path = chunk_dir / "nlu.yml"
        if not nlu_path.exists():
            nlu_prompt = tmpl_nlu.replace("{{ DIALOGUES }}", dialogues)
            nlu_yaml   = call_llama(
                args.base_url, args.model_name,
                [{"role": "system", "content": SYSTEM_MSG},
                 {"role": "user",   "content": nlu_prompt}],
                args.max_tokens, args.temperature,
            )
            nlu_path.write_text(nlu_yaml, encoding="utf-8")
            print(f"[WRITE] {nlu_path.relative_to(out_root.parent)}")
            time.sleep(args.sleep)
        else:
            nlu_yaml = nlu_path.read_text(encoding="utf-8")

        # -------------------------------------------------------------------#
        # 2) DOMAIN ----------------------------------------------------------
        # -------------------------------------------------------------------#
        domain_path = chunk_dir / "domain.yml"
        if not domain_path.exists():
            domain_prompt = (
                tmpl_domain
                .replace("{{ NLU }}", nlu_yaml.strip())
                .replace("{{ DIALOGUES }}", dialogues)
            )
            domain_yaml = call_llama(
                args.base_url, args.model_name,
                [{"role": "system", "content": SYSTEM_MSG},
                 {"role": "user",   "content": domain_prompt}],
                args.max_tokens, args.temperature,
            )
            domain_path.write_text(domain_yaml, encoding="utf-8")
            print(f"[WRITE] {domain_path.relative_to(out_root.parent)}")
            time.sleep(args.sleep)
        else:
            domain_yaml = domain_path.read_text(encoding="utf-8")

        # -------------------------------------------------------------------#
        # 3) STORIES ---------------------------------------------------------
        # -------------------------------------------------------------------#
        stories_path = chunk_dir / "stories.yml"
        if not stories_path.exists():
            stories_prompt = (
                tmpl_stories
                .replace("{{ NLU }}",    nlu_yaml.strip())
                .replace("{{ DOMAIN }}", domain_yaml.strip())
                .replace("{{ DIALOGUES }}", dialogues)
            )
            stories_yaml = call_llama(
                args.base_url, args.model_name,
                [{"role": "system", "content": SYSTEM_MSG},
                 {"role": "user",   "content": stories_prompt}],
                args.max_tokens, args.temperature,
            )
            stories_path.write_text(stories_yaml, encoding="utf-8")
            print(f"[WRITE] {stories_path.relative_to(out_root.parent)}")
            time.sleep(args.sleep)

    print("All done.")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
