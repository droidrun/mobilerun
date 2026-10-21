import logging
from dataclasses import dataclass

from mobilerun.orchestration.events import (
    make_agent_event_callback,
    raw_agent_event_to_orchestration_event,
)


logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)s:%(name)s:%(message)s",
)


@dataclass
class FakeTapEvent:
    action: str = "tap"
    x: int = 120
    y: int = 300
    description: str = "Tapped login button"


received_events = []


def external_callback(event):
    print("CALLBACK RECEIVED:", event)
    received_events.append(event)


raw_event = FakeTapEvent()

normalized = raw_agent_event_to_orchestration_event(
    raw_event,
    task_id="task-123",
)

print("NORMALIZED EVENT:")
print(normalized)

agent_callback = make_agent_event_callback(
    task_id="task-123",
    event_callback=external_callback,
    use_cli_handler=False,
)

agent_callback(raw_event)

assert len(received_events) == 1
assert received_events[0].task_id == "task-123"
assert received_events[0].payload["action"] == "tap"
assert received_events[0].payload["x"] == 120
assert received_events[0].payload["y"] == 300

print("events.py manual test passed")