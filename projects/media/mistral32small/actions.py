# actions.py
from __future__ import annotations
from typing import Any, Dict, Text, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

# --- tiny fake catalogue for demo purposes ---
FAKE_HOTELS = {
    # city: list of (name, district)
    "lyon": [
        ("Sofitel Royal", "centre"),
        ("Mercure Lumière", "monplaisir"),
        ("Ibis Lyon Bron", "bron"),
    ],
    "paris": [
        ("Ibis Bastille", "bastille"),
        ("Mercure Montparnasse", "montparnasse"),
    ],
}

def _get_slot(tracker: Tracker, name: str, default: Optional[str] = None) -> Optional[str]:
    val = tracker.get_slot(name)
    if isinstance(val, str):
        return val.strip()
    return default

class ActionSearchHotels(Action):
    def name(self) -> Text:
        return "action_search_hotels"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        city = (_get_slot(tracker, "city") or "").lower()
        district = (_get_slot(tracker, "district") or "").lower()

        matches = []
        if city in FAKE_HOTELS:
            for (name, d) in FAKE_HOTELS[city]:
                if not district or district == d:
                    matches.append((name, d))

        if matches:
            # simple textual list; you could also set another slot with the list
            lines = [f"- {name} ({d})" for (name, d) in matches[:5]]
            dispatcher.utter_message(text="Voici des hôtels disponibles :\n" + "\n".join(lines))
            return [SlotSet("search_status", "found")]
        else:
            dispatcher.utter_message(text="Désolé, aucun hôtel ne correspond à votre recherche.")
            return [SlotSet("search_status", "none")]

class ActionCheckAvailability(Action):
    def name(self) -> Text:
        return "action_check_availability"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        # naive “availability” rule for prototype:
        # if nights <= 5 and rooms <= 3 → yes, else no
        nights = tracker.get_slot("nights")
        rooms = tracker.get_slot("num_rooms")

        try:
            n_nights = int(nights) if nights is not None else 1
        except Exception:
            n_nights = 1
        try:
            n_rooms = int(rooms) if rooms is not None else 1
        except Exception:
            n_rooms = 1

        is_ok = (n_nights <= 5 and n_rooms <= 3)
        if is_ok:
            dispatcher.utter_message(text="Bonne nouvelle, des chambres sont disponibles.")
            return [SlotSet("availability", "yes")]
        else:
            dispatcher.utter_message(text="Malheureusement, ce n'est pas disponible aux dates demandées.")
            return [SlotSet("availability", "no")]

class ActionProvideAmenities(Action):
    def name(self) -> Text:
        return "action_provide_amenities"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        # demo reply
        dispatcher.utter_message(text="Piscine: non · Jacuzzi: non · Animaux: acceptés à Ibis Lyon Bron.")
        return []

class ActionProvidePrice(Action):
    def name(self) -> Text:
        return "action_provide_price"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Exemples de tarifs: simple 62€ · double 110€ · supérieur 172€ (variables selon dates).")
        return []
