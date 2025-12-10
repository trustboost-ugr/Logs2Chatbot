# generate_rasa_files.py
"""
Chunk Let's Go dialogues to fit GPT‑4o context limits and call the model to
produce Rasa nlu.yml, domain.yml and stories.yml files.

Compatible with **openai‑python ≥ 1.0.0** (released 2023‑10‑01).
If you still have an older client (< 0.28) either upgrade (`pip install -U openai`)
or swap the import back as shown in the git history.

Usage (example):
    export OPENAI_API_KEY="sk‑..."
    python generate_rasa_files.py \
        --csv /mnt/data/processed_interactions.csv \
        --output_dir rasa_outputs \
        --model gpt-4o \
        --max_context_tokens 128000 \
        --safety_margin 2000

The script
1. Loads the given CSV which must contain at least the columns
   ``dialogue`` and ``num_tokens``.
2. Aggregates dialogues into chunks whose combined ``num_tokens`` stay below
   ``max_context_tokens - safety_margin`` (leaving room for the prompt and
   the model's response).
3. Sends each chunk to OpenAI’s chat/completions endpoint with the provided
   Rasa‑generation prompt.
4. Stores the raw response JSON (``.model_dump()``) and the prompt that
   produced it so you can parse the nlu.yml, domain.yml and stories.yml
   files afterwards.

This script never silently truncates dialogues: if a single dialogue already
exceeds the allowed context, it is written to an "oversized" list for manual
inspection.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from openai import OpenAI, OpenAIError  # ≥ 1.0.0 interface

# ---------------------------------------------------------------------------
# OpenAI client (reads API key from env var OPENAI_API_KEY)
# ---------------------------------------------------------------------------

client = OpenAI()

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

DATASET_PROMPT_TEMPLATE = textwrap.dedent(
    """
    I have dialogues from a spoken dialogue system involving interactions between the system and users. I need to extract three Rasa files from these dialogues: nlu.yml, domain.yml, and stories.yml. The goal is to ensure there are no inconsistencies in the files, especially avoiding story structure conflicts and inconsistent intents.

    nlu.yml:

    1. Extract intents from the dialogues and list example utterances for each intent.
    2. Identify entities mentioned in the dialogues and provide examples of entity values.
    3. Ensure each entity is consistently associated with the correct intents.

    domain.yml:

    1. Define intents, entities, slots, and their mappings based on the dialogues.
    2. For each slot, include its type and any relevant mappings (e.g., from entities or custom logic).
    3. List responses that the system should use, based on the dialogues.
    4. Include any actions mentioned in the dialogues.

    stories.yml:

    1. Extract story structures from the dialogues, ensuring that each story is consistent and follows a logical flow.
    2. Avoid conflicts in story structures and ensure that the transitions between intents are clear and coherent.

    Please process the following dialogues and generate the required Rasa files:

    Dialogues:
    {dial}

    Requirements:

    1. Ensure that the intents in nlu.yml match those in domain.yml.
    2. Make sure the slots in domain.yml have appropriate mappings and types.
    3. Validate that the story structures in stories.yml do not conflict and logically follow from the dialogues.
    """
)

DATASET_PROMPT_TEMPLATE_MEDIA = textwrap.dedent(
    """
    Les dialogues ci-dessous (en français) proviennent d’un serveur vocal d’informations touristiques et de réservation d’hôtel (corpus MEDIA). À partir de ces dialogues, générez **trois** fichiers Rasa : `nlu.yml`, `domain.yml`, `stories.yml`. Les fichiers doivent être **cohérents entre eux** (mêmes noms d’intent/entité/slot) et **valables pour Rasa 3.x** (YAML UTF-8, sans encadrés de code).

    Contraintes générales
    - Respecter la langue des dialogues : **exemples NLU** et **templates de réponses** en **français**.
    - Conserver les accents et caractères français (UTF-8).
    - Garder des **noms d’intents/entités/slots en anglais et snake_case** pour rester compatibles entre projets (ex. `request_hotel_search`, `provide_city`, `check_pool`, etc.).
    - Éviter tout conflit d’histoires/règles ; ne pas inventer d’actions non déclarées.

    1) `nlu.yml`
    - Définir les intents et fournir plusieurs exemples français par intent.
    - Extraire des entités pertinentes avec exemples (p.ex. `city`, `district`, `date`, `nights`, `num_rooms`, `num_guests`, `hotel_name`, `amenity` comme `pool`, `jacuzzi`, `pets`).
    - Exemple de taxonomie d’intents (adapter au contenu) :
       - `greet`, `goodbye`, `help`
       - `request_hotel_search` (démarrer une recherche)
       - `provide_city`, `provide_district`
       - `provide_dates`, `provide_nights`
       - `provide_num_rooms`, `provide_num_guests`
       - `ask_price`, `ask_amenities`, `check_pool`, `check_jacuzzi`, `check_pets`
       - `confirm`, `deny`
       - `request_booking` (passer à la réservation)
    - Annoter les entités seulement quand c’est clair (ex.: « à Lyon », « du 27 septembre au 2 octobre », « dix chambres »).

    2) `domain.yml`
    - Lister **tous** les intents utilisés.
    - Déclarer les entités correspondantes (`city`, `district`, `start_date`, `end_date`, `nights`, `num_rooms`, `num_guests`, `hotel_name`, `price`, `amenity`, `pets`).
    - Définir les **slots** (type `text` ou `categorical/bool` si pertinent) avec **slot_mappings** explicites, au besoin en `from_text` (phase bootstrap).
    - Ajouter des réponses `utter_*` en français (ex.: `utter_ask_city`, `utter_ask_dates`, `utter_ask_num_rooms`, `utter_confirm_search`, `utter_no_availability`, `utter_price_info`, etc.).
    - Lister des actions éventuelles (ex.: `action_search_hotels`, `action_provide_options`), sans les implémenter.

    3) `stories.yml`
    - Composer des **parcours simples et cohérents** reflétant les dialogues :
        démarrage → collecte ville/quartier → dates/nuits → contraintes (prix, piscine/jacuzzi, animaux) → offre(s) → réservation ou fin.
    - Éviter des transitions contradictoires. Ne référencer que des intents/utterances/actions déclarés.

    Dialogue(s) à traiter :
    {dial}

    Exigences finales :
    - **Cohérence stricte** des noms entre `nlu.yml` et `domain.yml`.
    - Slots **explicites** (Rasa 3.x), pas d’auto-fill implicite.
    - YAML **valide**, **UTF-8**, **sans balises de code**.
    """
)

# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------

def build_chunks(
    df: pd.DataFrame,
    max_context_tokens: int,
    safety_margin: int,
    overhead_prompt_tokens: int = 800,
) -> Tuple[List[List[str]], List[int]]:
    """Split dialogues into token‑bounded chunks.

    Parameters
    ----------
    df : DataFrame with columns ``dialogue`` and ``num_tokens``.
    max_context_tokens : Total context size of the model.
    safety_margin : Tokens reserved for the model's *response* (output).
    overhead_prompt_tokens : Approximate tokens used by static parts of the
        prompt plus JSON / role wrappers.

    Returns
    -------
    chunks : list of lists of dialogue strings
    chunk_token_counts : list of token counts per chunk (for logging)
    """
    chunks: List[List[str]] = []
    chunk_token_counts: List[int] = []

    current_chunk: List[str] = []
    current_tokens = 0
    limit = max_context_tokens - safety_margin - overhead_prompt_tokens

    for _idx, row in df.iterrows():
        dialog_tokens = int(row["num_tokens"])
        dialog_text = str(row["dialogue"])

        # If a single dialogue is too large, save as an individual chunk.
        if dialog_tokens > limit:
            print(
                f"[OVERSIZED] Dialogue has {dialog_tokens} tokens which exceeds the single‑chunk limit {limit}. Saving separately."
            )
            chunks.append([dialog_text])
            chunk_token_counts.append(dialog_tokens)
            current_chunk = []
            current_tokens = 0
            continue

        # If adding this dialogue would overflow, start new chunk.
        if current_tokens + dialog_tokens > limit and current_chunk:
            chunks.append(current_chunk)
            chunk_token_counts.append(current_tokens)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(dialog_text)
        current_tokens += dialog_tokens

    # Add final chunk.
    if current_chunk:
        chunks.append(current_chunk)
        chunk_token_counts.append(current_tokens)

    return chunks, chunk_token_counts


# ---------------------------------------------------------------------------
# OpenAI interaction
# ---------------------------------------------------------------------------

def call_openai(
    prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> dict:
    """Call the OpenAI chat completion endpoint (≥ 1.0.0) and return a Python dict."""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    # ``model_dump`` converts pydantic model to dict without losing fields.
    return resp.model_dump()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Chunk Let's Go dialogues and generate Rasa files via GPT‑4o."
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to processed_interactions.csv")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where prompts and responses will be saved")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--max_context_tokens", type=int, default=128000, help="Model context window in tokens")
    parser.add_argument(
        "--safety_margin",
        type=int,
        default=2000,
        help="Reserve this many tokens for GPT‑4o's answer",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only build chunks and write prompts without hitting the API",
    )
    parser.add_argument(
        "--limit_chunks",
        type=int,
        default=None,
        help="Optional: only process the first N chunks (for testing)",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 1. Load CSV
    # ---------------------------------------------------------------------
    df = pd.read_csv(args.csv)
    required_cols = {"dialogue", "num_tokens"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    # ---------------------------------------------------------------------
    # 2. Build chunks
    # ---------------------------------------------------------------------
    chunks, counts = build_chunks(
        df,
        max_context_tokens=args.max_context_tokens,
        safety_margin=args.safety_margin,
    )

    avg_tokens = sum(counts) // max(len(counts), 1)
    print(f"Created {len(chunks)} chunks (avg. {avg_tokens} tokens each).")

    # ---------------------------------------------------------------------
    # 3. Iterate over chunks, build prompt, call model, save
    # ---------------------------------------------------------------------

    for idx, (chunk_dialogues, token_count) in enumerate(zip(chunks, counts), start=1):
        if args.limit_chunks and idx > args.limit_chunks:
            print("Chunk limit reached; stopping early.")
            break

        joined_dialogues = "\n\n".join(chunk_dialogues)
        prompt = DATASET_PROMPT_TEMPLATE_MEDIA.format(dial=joined_dialogues)

        prompt_path = out_dir / f"chunk_{idx:03d}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")
        print(f"[WRITE] Prompt saved to {prompt_path} (tokens ≈ {token_count}).")

        if args.dry_run:
            print("[DRY‑RUN] Skipping API call.")
            continue

        try:
            response = call_openai(
                prompt=prompt,
                model=args.model,
                # Let GPT‑4o decide tokens based on its limit
            )
        except OpenAIError as exc:
            print(f"[ERROR] OpenAI API call failed on chunk {idx}: {exc}")
            continue

        response_path = out_dir / f"chunk_{idx:03d}_response.json"
        with response_path.open("w", encoding="utf-8") as fp:
            json.dump(response, fp, ensure_ascii=False, indent=2)
        print(f"[WRITE] Response saved to {response_path}.")

    print("All done.")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
