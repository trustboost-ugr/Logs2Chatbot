#!/usr/bin/env python
import argparse
import json
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple


def load_conversations(path: str) -> List[Dict[str, Any]]:
    """Load a list of conversations from a harness log.

    Supports:
      - top-level list of convos
      - dict with a list under 'convos' / 'conversations' / 'dialogs' / 'probes'
      - dict mapping ids -> convo objects
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Case 1: already a list of conversations
    if isinstance(data, list):
        return data

    # Case 2: dict with a list-valued field
    if isinstance(data, dict):
        for key in ["convos", "conversations", "dialogs", "probes", "sessions"]:
            if key in data and isinstance(data[key], list):
                return data[key]

        # Case 3: mapping from ids to convo dicts
        if all(isinstance(v, dict) for v in data.values()):
            return list(data.values())

        # Fallback: treat the whole object as a single conversation
        return [data]

    # Fallback for unexpected shapes
    return []


def _get_first_numeric(d: Dict[str, Any], keys: List[str]) -> Optional[float]:
    """Try to extract the first numeric value among the given keys."""
    for key in keys:
        if key in d:
            val = d[key]
            if isinstance(val, (int, float)):
                return float(val)
    return None


def _get_first_bool(d: Dict[str, Any], keys: List[str]) -> Optional[bool]:
    """Try to extract a boolean value among the given keys."""
    for key in keys:
        if key in d:
            val = d[key]
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return bool(val)
            if isinstance(val, str):
                low = val.strip().lower()
                if low in {"true", "yes", "y", "1", "completed", "success"}:
                    return True
                if low in {"false", "no", "n", "0"}:
                    return False
    return None


def get_conv_completion(conv: Dict[str, Any]) -> bool:
    """Heuristic: decide whether a conversation is complete."""
    # direct fields
    completed = _get_first_bool(conv, ["completed", "success", "is_complete", "done"])
    if completed is not None:
        return completed

    # nested metrics (common pattern: conv["kpis"]["completed"])
    for key in ["kpis", "metrics", "meta"]:
        if key in conv and isinstance(conv[key], dict):
            completed = _get_first_bool(conv[key], ["completed", "success", "is_complete", "done"])
            if completed is not None:
                return completed

    # default: not complete
    return False


def get_conv_turns(conv: Dict[str, Any]) -> int:
    """Heuristic: number of turns (user+bot messages) in a conversation."""
    # try direct numeric KPI fields
    turns = _get_first_numeric(conv, ["turns", "num_turns", "n_turns", "length"])
    if turns is not None:
        return int(round(turns))

    # nested metrics (kpis, meta)
    for key in ["kpis", "metrics", "meta"]:
        if key in conv and isinstance(conv[key], dict):
            turns = _get_first_numeric(conv[key], ["turns", "num_turns", "n_turns", "length"])
            if turns is not None:
                return int(round(turns))

    # count messages if present
    for key in ["messages", "events", "history"]:
        if key in conv and isinstance(conv[key], list):
            return len(conv[key])

    # fallback: 0 (should be rare; can be inspected in the raw JSON)
    return 0


def is_help_probe_index(idx_zero_based: int) -> bool:
    """Decide if a probe is a help-detour probe based on its index.

    The harness injects a help detour every 5th probe.
    With 0-based indexing, this corresponds to indices 4, 9, 14, ...
    """
    return (idx_zero_based + 1) % 5 == 0


def compute_kpis_for_file(path: str) -> Tuple[int, float, float, float]:
    """Compute N, completion %, median turns, and help recovery % for one log file."""
    convos = load_conversations(path)
    n = len(convos)

    if n == 0:
        return 0, 0.0, 0.0, 0.0

    completion_flags: List[bool] = []
    turns_list: List[int] = []
    help_completion_flags: List[bool] = []

    for idx, conv in enumerate(convos):
        completed = get_conv_completion(conv)
        turns = get_conv_turns(conv)

        completion_flags.append(completed)
        turns_list.append(turns)

        if is_help_probe_index(idx):
            help_completion_flags.append(completed)

    compl_pct = 100.0 * sum(completion_flags) / n

    if turns_list:
        med_turns = float(statistics.median(turns_list))
    else:
        med_turns = 0.0

    if help_completion_flags:
        help_rec_pct = 100.0 * sum(help_completion_flags) / len(help_completion_flags)
    else:
        help_rec_pct = 0.0

    return n, compl_pct, med_turns, help_rec_pct


def infer_corpus_from_filename(filename: str) -> str:
    """Infer corpus (letsgo/media) from filename."""
    lower = filename.lower()
    if "letsgo" in lower or "lets_go" in lower or "lg_" in lower:
        return "letsgo"
    if "media" in lower:
        return "media"
    return "unknown"


def infer_model_from_filename(filename: str) -> str:
    """Infer model name from filename (very simple heuristic)."""
    lower = filename.lower()
    if "gpt4o" in lower or "gpt-4o" in lower:
        return "gpt4o"
    if "claude" in lower:
        return "claude"
    if "gemini" in lower:
        return "gemini"
    if "mistral" in lower:
        return "mistral"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute dialog-level KPIs from harness logs.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more JSON log files (e.g., media_gpt4o_convos_*.json).",
    )
    args = parser.parse_args()

    # Group by corpus for pretty printing
    rows_by_corpus: Dict[str, List[Tuple[str, str, int, float, float, float]]] = {}

    for path in args.paths:
        filename = os.path.basename(path)
        corpus = infer_corpus_from_filename(filename)
        model = infer_model_from_filename(filename)

        n, compl_pct, med_turns, help_rec_pct = compute_kpis_for_file(path)

        rows_by_corpus.setdefault(corpus, []).append(
            (filename, model, n, compl_pct, med_turns, help_rec_pct)
        )

    for corpus, rows in rows_by_corpus.items():
        print(f"\n=== Corpus: {corpus} ===")
        print(
            f"{'File':34} {'Model':10} {'N':>4} {'Compl%':>7} {'MedTurns':>8} {'HelpRec%':>8}"
        )
        print("-" * 80)
        for filename, model, n, compl_pct, med_turns, help_rec_pct in sorted(rows):
            print(
                f"{filename:34} {model:10} {n:4d} {compl_pct:7.1f} {med_turns:8.1f} {help_rec_pct:8.1f}"
            )


if __name__ == "__main__":
    main()
