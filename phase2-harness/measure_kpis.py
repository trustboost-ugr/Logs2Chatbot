# measure_kpis.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import signal
import socket
import random
import string
import subprocess
from pathlib import Path
from statistics import median
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import requests
import pandas as pd

# =========================
# Config
# =========================

# ⚠️ Point each project to its **actual Rasa project root** (folder that has domain.yml)
# or to a child like "output/.../1" that itself includes domain.yml. We will NOT climb to parents.
PROJECTS = [
  #("letsgo_gpt4o",   "output/gpt4o_new/1"),
  #("letsgo_claude",  "output/claude_opus_new/1"),
  #("letsgo_gemini",  "output/letsgo_gemini25_pro/1"),
  #("letsgo_mistral", "output/letsgo_mistral24b_32_seq/chunk_001"),
  #("media_claude",   "output/media_claude_opus_small/1"),
  #("media_gemini",   "output/media_gemini25_pro/1"),
  #("media_gpt4o",    "output/media_gpt4o_small/1"),
  #("media_mistral",  "output/media_mistral24b_seq/chunk_001"),
    ("letsgo_baseline", "output/baselines/letsgo"),
    ("media_baseline",  "output/baselines/media"),
]

BASE_PORT = 5005
SERVER_BOOT_TIMEOUT = 60
REQ_TIMEOUT = 20
TURN_BUDGET = 20            # max user replies per probe
BOT_TURN_CAP = 4            # max bot messages we record per turn
N_PROBES = 50
RANDOM_SEED = 42

RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")

LOG_DIR = Path("kpi_logs")
LOG_DIR.mkdir(exist_ok=True)

CSV_OUT = Path(f"kpi_results_{RUN_ID}.csv")
TEX_OUT = Path(f"kpi_table_{RUN_ID}.tex")

# =========================
# Utilities
# =========================

def find_free_port(start_from: int) -> int:
    port = start_from
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1

def has_domain(p: Path) -> bool:
    return (p / "domain.yml").exists() or (p / "domain.yaml").exists()

def resolve_project_root_strict(path_like: str) -> Optional[Path]:
    """
    Strict resolver:
      - Accept the provided path if it contains domain.yml
      - Else look in *children* that are commonly used ('.', '1', 'chunk_001', 'models/latest', etc.)
      - Do NOT climb to parents (prevents loading a *different* bot by mistake).
    """
    p = Path(path_like).resolve()
    if p.is_file():
        p = p.parent
    if not p.exists():
        return None

    # 1) Provided path directly
    if p.is_dir() and has_domain(p):
        return p

    # 2) Obvious children
    candidates = [
        p / ".",
        p / "1",
        p / "chunk_001",
        p / "models",
        p / "model",
        p / "latest",
    ]
    # Sometimes models/ contains a copy of the project; check one level below as well
    more = []
    for c in list(candidates):
        if c.is_dir():
            for child in c.iterdir():
                if child.is_dir():
                    more.append(child)
    candidates.extend(more)

    for q in candidates:
        try:
            if q.is_dir() and has_domain(q):
                return q.resolve()
        except Exception:
            pass

    return None

def check_dir_permissions(d: Path) -> None:
    if not os.access(str(d), os.R_OK):
        raise PermissionError(f"Directory not readable: {d}")
    if not os.access(str(d), os.X_OK):
        raise PermissionError(f"Directory not searchable (+x missing): {d}  ->  fix with: chmod u+X '{d}'")

def spawn_rasa_server(project_dir: Path, port: int) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "rasa", "run",
        "--enable-api",
        "--port", str(port),
        "--cors", "*",
        "--quiet",
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(project_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True
    )

def wait_until_ready(port: int, timeout: int) -> bool:
    url = f"http://127.0.0.1:{port}/status"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.6)
    return False

def get_domain(port: int) -> dict:
    """
    Try to fetch the domain as JSON. If the server doesn't support it (406),
    fall back to plain text (YAML). On fallback we return {'_raw_domain': <text>}.
    """
    url = f"http://127.0.0.1:{port}/domain"

    # Preferred: JSON
    try:
        r = requests.get(
            url,
            params={"format": "json"},
            headers={"Accept": "application/json"},
            timeout=REQ_TIMEOUT,
        )
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                pass  # will try fallback below
    except requests.RequestException:
        pass

    # Fallback: plain text (usually YAML)
    r = requests.get(url, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return {"_raw_domain": r.text}

def verify_domain_signature(port: int, model_key: str) -> None:
    """
    Ensure the loaded server matches the intended bot.
    For Let's Go: expect schedule_form OR utter_find_bus.
    For Media:    expect booking_form OR utter_ask_city/utter_searching_hotels.

    Works with JSON or plain-text (YAML) domain payloads.
    """
    dom = get_domain(port)

    is_letsgo_expected = model_key.startswith("letsgo_")
    is_media_expected  = model_key.startswith("media_")

    letsgo_ok = False
    media_ok  = False

    if "_raw_domain" in dom:
        raw = (dom["_raw_domain"] or "").lower()
        # substring checks for YAML/text
        letsgo_ok = ("schedule_form" in raw) or ("utter_find_bus" in raw) or ("utter_find_next_bus" in raw)
        media_ok  = ("booking_form" in raw) or ("utter_ask_city" in raw) or ("utter_searching_hotels" in raw)
    else:
        # JSON structure (Rasa >= 3 supports ?format=json)
        actions   = set(dom.get("actions") or [])
        forms_raw = dom.get("forms")
        if isinstance(forms_raw, dict):
            forms = set(forms_raw.keys())
        else:
            forms = set(forms_raw or [])
        responses = set((dom.get("responses") or {}).keys())

        letsgo_ok = ("schedule_form" in forms) or ("utter_find_bus" in responses) or ("utter_find_next_bus" in responses)
        media_ok  = ("booking_form" in forms) or ("utter_ask_city" in responses) or ("utter_searching_hotels" in responses)

    if is_letsgo_expected and not letsgo_ok:
        raise RuntimeError(f"{model_key}: server domain doesn't look like Let's Go (missing schedule_form/utter_find_bus).")
    if is_media_expected and not media_ok:
        raise RuntimeError(f"{model_key}: server domain doesn't look like Media (missing booking_form/utter_ask_city).")

def warmup_ping(port: int, model_key: str, tries: int = 5, delay: float = 0.8, allow_silent: bool = False) -> bool:
    conv_id = f"warmup_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    corpus = "letsgo" if model_key.startswith("letsgo_") else "media"

    if corpus == "letsgo":
        probes = [
            "hello",
            "help",
            "I need the 61A from downtown to the airport at 7 pm",
            "when is the next 28X from oakland to the airport?",
        ]
    else:
        probes = [
            "bonjour",
            "aide",
            "je veux réserver un hôtel à Paris du 3 au 5 octobre pour deux adultes",
            "je cherche un hôtel à Lyon avec piscine",
        ]

    for t in range(tries):
        text = probes[min(t, len(probes)-1)]
        try:
            rs = rest_send(port, conv_id, text)
            if any(bool(m.get("text") or m.get("image") or m.get("buttons")) for m in rs):
                return True
        except requests.RequestException:
            pass
        time.sleep(delay)

    return allow_silent

def stop_process_tree(proc: subprocess.Popen):
    try:
        if proc.poll() is None:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass

def rand_id(prefix: str) -> str:
    return prefix + "_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=10))

def corpus_from_model_name(model_key: str) -> str:
    return "letsgo" if model_key.startswith("letsgo_") else "media"

def lang_from_corpus(corpus: str) -> str:
    return "en" if corpus == "letsgo" else "fr"

# =========================
# Probes
# =========================

LETSGO_ORIGINS = [
    "downtown", "oakland", "shadyside", "squirrel hill", "greenfield",
    "south side", "north shore", "bloomfield", "lawrenceville", "strip district"
]
LETSGO_DESTS = [
    "the airport", "university of pittsburgh", "carnegie mellon", "station square",
    "waterfront", "homestead", "east liberty", "north hills", "south hills", "downtown"
]
LETSGO_TIMES = [
    "5 a.m.", "6:30 am", "7:15 in the morning", "noon", "2 pm",
    "3:45 pm", "6 pm", "7:30 in the evening", "9 pm", "10:15 pm"
]
LETSGO_ROUTES = ["28X", "61A", "54C", "71B", "64"]

MEDIA_CITIES = [
    "Paris", "Lyon", "Marseille", "Nice", "Bordeaux",
    "Toulouse", "Nantes", "Lille", "Strasbourg", "Montpellier"
]
MEDIA_AMENITIES = ["piscine", "jacuzzi", "parking", "animal accepté", "wifi"]
MEDIA_DATES = [
    "du 3 au 5 octobre", "du 10 au 12 novembre", "du 15 au 18 septembre",
    "du 20 au 22 juin", "du 1er au 3 août"
]
MEDIA_ROOM_TYPES = ["chambre double", "chambre simple"]
MEDIA_NUM_GUESTS = ["deux adultes", "trois adultes", "un adulte", "deux adultes et un enfant"]

def build_letsgo_case(i: int) -> Dict[str, str]:
    return {
        "route": LETSGO_ROUTES[i % len(LETSGO_ROUTES)],
        "origin": LETSGO_ORIGINS[i % len(LETSGO_ORIGINS)],
        "destination": LETSGO_DESTS[(i * 2) % len(LETSGO_DESTS)],
        "time": LETSGO_TIMES[(i * 3) % len(LETSGO_TIMES)],
    }

def build_media_case(i: int) -> Dict[str, str]:
    return {
        "city": MEDIA_CITIES[i % len(MEDIA_CITIES)],
        "dates": MEDIA_DATES[(i * 2) % len(MEDIA_DATES)],
        "amenity": MEDIA_AMENITIES[(i * 3) % len(MEDIA_AMENITIES)],
        "room_type": MEDIA_ROOM_TYPES[i % len(MEDIA_ROOM_TYPES)],
        "guests": MEDIA_NUM_GUESTS[i % len(MEDIA_NUM_GUESTS)],
    }

def initial_utterance(case: Dict[str, str], corpus: str, i: int) -> str:
    if corpus == "letsgo":
        r, o, d, t = case["route"], case["origin"], case["destination"], case["time"]
        templates = [
            f"I need the {r} from {o} to {d} at {t}",
            f"When is the next {r} from {o} to {d}?",
            f"I want a bus from {o} to {d} around {t}",
            f"Schedule for the {r} from {o} to {d} at {t}",
            f"From {o} to {d} at {t} on the {r}",
        ]
        return templates[i % len(templates)]
    else:
        city, dates, amen, room, guests = case["city"], case["dates"], case["amenity"], case["room_type"], case["guests"]
        templates = [
            f"Je cherche un hôtel à {city} {dates} pour {guests}",
            f"Je veux réserver un hôtel à {city} {dates}, {room}, {amen}",
            f"Je veux un hôtel à {city} {dates} avec {amen}",
            f"Un hôtel à {city} {dates} pour {guests}, {room}",
            f"Je veux réserver {room} à {city} {dates}",
        ]
        return templates[i % len(templates)]

# =========================
# Per-project completion schemas (alias groups)
# =========================

PROJECT_REQUIRED_SLOT_ALIASES: Dict[str, List[set]] = {
    # Let’s Go variants
    "letsgo_gpt4o": [
        {"location", "origin", "from", "departure"},
        {"destination", "to", "arrival"},
        {"time", "departure_time", "leaving_time"},
    ],
    "letsgo_claude": [
        {"departure", "origin", "from", "location"},
        {"destination", "to", "arrival"},
        {"departure_time", "time"},
    ],
    "letsgo_gemini": [
        {"origin", "from", "location", "departure"},
        {"destination", "to", "arrival"},
        {"time", "departure_time"},
    ],
    "letsgo_mistral": [
        {"departure", "departure_stop", "from", "origin"},
        {"destination", "destination_stop", "to", "arrival"},
        {"time", "departure_time"},
    ],
    # MEDIA variants (hotel)
    "media_gpt4o": [
        {"city", "ville", "destination_city"},
        {"dates", "date", "arrival_date", "departure_date", "checkin", "checkout"},
        {"num_guests", "guests", "nb_personnes", "people", "adults_children", "room_count", "room_type"},
    ],
    "media_claude": [
        {"city", "ville"},
        {"start_date", "end_date", "dates", "date", "arrival_date", "departure_date", "checkin", "checkout"},
        {"num_guests", "guests", "nb_personnes", "people", "room_type"},
    ],
    "media_gemini": [
        {"city", "ville"},
        {"dates", "date", "arrival_date", "departure_date", "checkin", "checkout",
         "start_date", "end_date", "date_reference"},  # <-- add these
        {"num_guests", "guests", "nb_personnes", "people", "room_type"},
    ],
    "media_mistral": [
        {"city", "ville"},
        {"dates", "date", "arrival_date", "departure_date", "checkin", "checkout"},
        # consider either adults OR rooms as the “3rd” requirement
        {"number_of_adults", "num_guests", "guests", "nb_personnes", "people", "number_of_rooms", "number_of_nights"},
    ],
}

# =========================
# Project-specific success detection (media_gpt4o ONLY)
# =========================

MEDIA_GPT4O_REQUIRED_SLOTS = {"city", "start_date", "end_date", "num_guests"}
MEDIA_GPT4O_SUCCESS_ACTIONS = {"utter_hotel_detail", "utter_confirm_search", "action_search_hotels"}

def non_empty(v) -> bool:
    return v not in (None, "", [], {}, False)

def tracker_success_media_gpt4o(tracker: dict) -> bool:
    if not isinstance(tracker, dict):
        return False
    active_loop = tracker.get("active_loop")
    if isinstance(active_loop, dict):
        active_loop = active_loop.get("name")
    loop_off = (not active_loop)
    slots = tracker.get("slots") or {}
    have_slots = all(non_empty(slots.get(s)) for s in MEDIA_GPT4O_REQUIRED_SLOTS)

    recent_actions = []
    for ev in tracker.get("events", []):
        if ev.get("event") == "action":
            recent_actions.append(ev.get("name"))
    hit_submit = bool(set(recent_actions[-6:]) & MEDIA_GPT4O_SUCCESS_ACTIONS)

    return loop_off and have_slots and hit_submit

# =========================
# Completion checks (default / alias-based)
# =========================

def project_complete_alias_based(project_key: str, tracker: dict) -> bool:
    slots = tracker.get("slots", {}) or {}
    groups = PROJECT_REQUIRED_SLOT_ALIASES.get(project_key, [])
    for group in groups:
        if not any((k in slots and non_empty(slots[k])) for k in group):
            return False
    return True

def project_complete(project_key: str, tracker: dict) -> bool:
    if project_key == "media_gpt4o":
        return tracker_success_media_gpt4o(tracker)
    return project_complete_alias_based(project_key, tracker)

# =========================
# HTTP helpers
# =========================

def rest_send(port: int, conv_id: str, text: str) -> List[dict]:
    url = f"http://127.0.0.1:{port}/webhooks/rest/webhook"
    payload = {"sender": conv_id, "message": text}
    r = requests.post(url, json=payload, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.json()

def get_tracker(port: int, conv_id: str) -> dict:
    url = f"http://127.0.0.1:{port}/conversations/{conv_id}/tracker"
    r = requests.get(url, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    return r.json()

# =========================
# Fallback / loop detection
# =========================

def consecutive_fallbacks(tracker: dict, n: int = 2) -> bool:
    events = tracker.get("events") or []
    count = 0
    for ev in reversed(events):
        if ev.get("event") == "action" and ev.get("name") == "action_default_fallback":
            count += 1
            if count >= n:
                return True
        elif ev.get("event") in {"action", "user"}:
            break
    return False

def repeated_bot_text(bot_history: List[str], repeat_threshold: int = 3) -> bool:
    if len(bot_history) < repeat_threshold:
        return False
    last = (bot_history[-1] or "").strip().lower()
    if not last:
        return False
    c = 1
    for t in reversed(bot_history[:-1]):
        if (t or "").strip().lower() == last:
            c += 1
            if c >= repeat_threshold:
                return True
        else:
            break
    return False

# =========================
# Slot value resolution
# =========================

def value_for_slot(project_key: str, slot_name: str, case: Dict[str, str], corpus: str) -> Optional[str]:
    s = (slot_name or "").lower()

    if corpus == "letsgo":
        # check TIME first so "departure_time" maps to the time value
        if "time" in s:
            return case["time"]
        if any(k in s for k in ["destination", "to", "arrival"]):
            return case["destination"]
        if any(k in s for k in ["origin", "from", "departure", "location", "depart"]):
            return case["origin"]
        if any(k in s for k in ["route", "line", "bus"]):
            return case["route"]
        if "confirm" in s:
            return "yes"
    else:
        if any(k in s for k in ["city", "ville", "destination_city"]):
            return case["city"]
        if any(k in s for k in ["date", "checkin", "checkout"]):
            return case["dates"]
        if any(k in s for k in ["guest", "personne", "people", "adults", "room_count"]):
            return case["guests"]
        if "room_type" in s or "chambre" in s:
            return case["room_type"]
        if any(k in s for k in ["amenity", "piscine", "jacuzzi", "parking", "wifi"]):
            return case["amenity"]
        if "confirm" in s:
            return "oui"
    return None

def heuristic_answer_for_bot_text(bot_texts: List[str], case: Dict[str, str], corpus: str) -> Optional[str]:
    joined = " ".join((t or "").lower() for t in bot_texts)

    if corpus == "letsgo":
        # prefer time first; many prompts start with "when"
        if ("time" in joined) or ("when" in joined) or ("what time" in joined) or ("depart" in joined):
            return case["time"]
        if ("to" in joined) or ("going" in joined) or ("destination" in joined) or ("where to" in joined):
            return case["destination"]
        if ("from" in joined) or ("leaving" in joined) or ("where from" in joined):
            return case["origin"]
        if ("route" in joined) or ("line" in joined) or ("bus" in joined):
            return case["route"]
        if ("yes/no" in joined) or ("confirm" in joined) or ("did i get that right" in joined):
            return "yes"
    else:
        if "ville" in joined or "city" in joined or "où" in joined or "où souhaitez-vous" in joined:
            return case["city"]
        if "date" in joined or "check-in" in joined or "arrivée" in joined or "départ" in joined or "séjour" in joined:
            return case["dates"]
        if "personne" in joined or "adulte" in joined or "occupant" in joined or "combien de" in joined:
            return case["guests"]
        if "chambre" in joined or "type" in joined:
            return case["room_type"]
        if "équipement" in joined or "commodité" in joined or "wifi" in joined or "piscine" in joined or "jacuzzi" in joined or "parking" in joined:
            return case["amenity"]
        if "confirmer" in joined or "oui/non" in joined:
            return "oui"
    return None

# =========================
# Conversational KPI run
# =========================

def run_probe_sequence(port: int, model_key: str, i: int) -> Tuple[bool, int, bool, dict]:
    conv_id = rand_id(model_key)
    corpus = corpus_from_model_name(model_key)
    lang = lang_from_corpus(corpus)
    random.seed(RANDOM_SEED + i)

    case = build_letsgo_case(i) if corpus == "letsgo" else build_media_case(i)
    opener = initial_utterance(case, corpus, i)

    help_probe = ((i % 5) == 4)
    help_token = "help" if lang == "en" else "aide"

    convo_log = {"conversation_id": conv_id, "steps": [], "help_probe": help_probe}
    completed = False
    turns_used = 0
    bot_text_history: List[str] = []

    # 1) Send opener
    bot_resps = rest_send(port, conv_id, opener)
    bot_texts = [m.get("text", "") for m in bot_resps if "text" in m][:BOT_TURN_CAP]
    bot_text_history.extend([t for t in bot_texts if t])
    convo_log["steps"].append({"user": opener, "bot": bot_texts})
    turns_used += 1 + (1 if bot_texts else 0)

    tracker = get_tracker(port, conv_id)
    if project_complete(model_key, tracker):
        completed = True
        convo_log["final_tracker"] = tracker
        convo_log["completed"] = completed
        convo_log["turns"] = turns_used
        return completed, turns_used, help_probe, convo_log

    # 2) Optional help detour
    if help_probe:
        bot_resps = rest_send(port, conv_id, help_token)
        bot_texts = [m.get("text", "") for m in bot_resps if "text" in m][:BOT_TURN_CAP]
        bot_text_history.extend([t for t in bot_texts if t])
        convo_log["steps"].append({"user": help_token, "bot": bot_texts})
        turns_used += 1 + (1 if bot_texts else 0)

        tracker = get_tracker(port, conv_id)
        if project_complete(model_key, tracker):
            completed = True
            convo_log["final_tracker"] = tracker
            convo_log["completed"] = completed
            convo_log["turns"] = turns_used
            return completed, turns_used, help_probe, convo_log

    # 3) Interactive loop
    for _ in range(TURN_BUDGET):
        tracker = get_tracker(port, conv_id)

        if project_complete(model_key, tracker):
            completed = True
            break

        if consecutive_fallbacks(tracker, n=2) or repeated_bot_text(bot_text_history, repeat_threshold=3):
            completed = False
            break

        active_loop = tracker.get("active_loop")
        if isinstance(active_loop, dict):
            active_loop = active_loop.get("name")
        requested_slot = tracker.get("slots", {}).get("requested_slot")
        reply = None
        if active_loop and requested_slot:
            reply = value_for_slot(model_key, str(requested_slot), case, corpus)

        if reply is None:
            last_bot_texts = convo_log["steps"][-1].get("bot", []) if convo_log["steps"] else []
            reply = heuristic_answer_for_bot_text(last_bot_texts, case, corpus)

        if reply is None:
            reply = "yes" if lang == "en" else "oui"

        bot_resps = rest_send(port, conv_id, reply)
        bot_texts = [m.get("text", "") for m in bot_resps if "text" in m][:BOT_TURN_CAP]
        bot_text_history.extend([t for t in bot_texts if t])
        convo_log["steps"].append({"user": reply, "bot": bot_texts})
        turns_used += 1 + (1 if bot_texts else 0)

        tracker = get_tracker(port, conv_id)
        if project_complete(model_key, tracker):
            completed = True
            break

    tracker = get_tracker(port, conv_id)
    if not completed:
        completed = project_complete(model_key, tracker)
    convo_log["final_tracker"] = tracker
    convo_log["completed"] = completed
    convo_log["turns"] = turns_used

    return completed, turns_used, help_probe, convo_log

# =========================
# Model measurement
# =========================

def measure_model(model_key: str, provided_path: str, base_port: int) -> Dict[str, float]:
    resolved = resolve_project_root_strict(provided_path)
    if resolved is None:
        raise RuntimeError(
            f"{model_key}: could not locate a Rasa project root inside '{provided_path}' "
            f"(need a folder that contains domain.yml)."
        )
    check_dir_permissions(resolved)

    port = find_free_port(base_port)
    print(f"[{model_key}] starting server on port {port} in {resolved}")
    proc = spawn_rasa_server(resolved, port)

    if not wait_until_ready(port, SERVER_BOOT_TIMEOUT):
        stop_process_tree(proc)
        raise RuntimeError(f"{model_key}: server not ready on {port} within {SERVER_BOOT_TIMEOUT}s")

    # Domain signature check — ensure the correct bot is actually loaded
    try:
        verify_domain_signature(port, model_key)
    except Exception as e:
        stop_process_tree(proc)
        raise

    # Warm up
    if not warmup_ping(port, model_key, allow_silent=True):
        print(f"[WARN] {model_key}: warm-up probes returned empty responses; continuing to run scripted probes.")

    random.seed(RANDOM_SEED)
    completed_list, turns_list, help_completed, logs = [], [], [], []

    for i in range(N_PROBES):
        try:
            comp, turns, help_probe, convo_log = run_probe_sequence(port, model_key, i)
            completed_list.append(1 if comp else 0)
            turns_list.append(turns)
            if help_probe:
                help_completed.append(1 if comp else 0)
            logs.append(convo_log)
        except Exception as e:
            completed_list.append(0)
            turns_list.append(TURN_BUDGET * 2)
            logs.append({"error": str(e), "conversation_id": rand_id(model_key)})

    # Save logs with meta + timestamped filename
    out_json = LOG_DIR / f"{model_key}_convos_{RUN_ID}.json"
    meta = {
        "run_id": RUN_ID,
        "model_key": model_key,
        "cwd": str(Path.cwd()),
        "project_dir": str(resolved.resolve()),
        "port": port,
        "n_probes": len(logs),
        "completed_count_in_memory": int(sum(1 for x in logs if x.get("completed"))),
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    payload = {"_meta": meta, "conversations": logs}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    abs_path = out_json.resolve()
    print(f"[{model_key}] wrote log to: {abs_path}")

    # Recount completions from disk and cross-check
    loaded = json.loads(Path(abs_path).read_text(encoding="utf-8"))
    disk_count = sum(1 for x in loaded.get("conversations", []) if x.get("completed"))
    mem_count  = meta["completed_count_in_memory"]

    if disk_count != mem_count:
        print(f"[WARN] {model_key}: completed_count mismatch  memory={mem_count}  disk={disk_count}  file={abs_path}")

    completion_rate = 100.0 * (disk_count / max(1, meta["n_probes"]))
    turns_median = int(median(turns_list)) if turns_list else 0
    help_recovery = 100.0 * (sum(help_completed) / len(help_completed)) if help_completed else 0.0

    print(f"[{model_key}] completion={completion_rate:.1f}%  median_turns={turns_median}  help_recovery={help_recovery:.1f}%")

    stop_process_tree(proc)

    return {
        "Model": model_key.replace("_", " "),
        "Completion (%)": round(completion_rate, 1),
        "Turns (median)": turns_median,
        "Help recovery (%)": round(help_recovery, 1),
    }

def write_outputs(rows: List[Dict[str, object]]):
    df = pd.DataFrame(rows, columns=["Model", "Completion (%)", "Turns (median)", "Help recovery (%)"])
    df.to_csv(CSV_OUT, index=False)

    order = [
        "letsgo gpt4o", "letsgo claude", "letsgo gemini", "letsgo mistral",
        "media gpt4o", "media claude", "media gemini", "media mistral"
    ]
    df["__order"] = df["Model"].str.lower().apply(lambda x: order.index(x) if x in order else 999)
    df_sorted = df.sort_values("__order").drop(columns="__order")

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Dialogue-level KPIs (median across 50 scripted probes).}")
    lines.append("\\label{tab:core-kpis}")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Completion (\\%)} & \\textbf{Turns (median)} & \\textbf{Help recovery (\\%)} \\\\")
    lines.append("\\midrule")
    for _, r in df_sorted.iterrows():
        short = (r["Model"].replace("letsgo ", "").replace("media ", "")).title()
        lines.append(f"{short} & {int(r['Completion (%)'])} & {int(r['Turns (median)'])} & {int(r['Help recovery (%)'])} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    TEX_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {CSV_OUT.resolve()} and {TEX_OUT.resolve()}")

def main():
    rows = []
    port_cursor = BASE_PORT
    print(f"[RUN] RUN_ID={RUN_ID}  CWD={Path.cwd().resolve()}")
    for model_key, provided_path in PROJECTS:
        try:
            rows.append(measure_model(model_key, provided_path, port_cursor))
        except Exception as e:
            print(f"[ERROR] {model_key}: {e}")
            rows.append({
                "Model": model_key.replace("_", " "),
                "Completion (%)": 0.0,
                "Turns (median)": TURN_BUDGET * 2,
                "Help recovery (%)": 0.0,
            })
        port_cursor += 1
    write_outputs(rows)

if __name__ == "__main__":
    main()
