from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

class ActionProvideSchedule(Action):
    def name(self) -> Text:
        return "action_provide_schedule"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        bus_route = tracker.get_slot("bus_route") or "any route"
        dep = tracker.get_slot("departure") or "your departure"
        dest = tracker.get_slot("destination") or "your destination"
        t = tracker.get_slot("departure_time") or "your time"
        dispatcher.utter_message(
            text=f"(stub) Looking up buses for {bus_route}, from {dep} to {dest} at {t}…"
        )
        return []
