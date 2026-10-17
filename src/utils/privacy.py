"""Keep processing local: switch off third-party usage analytics for this process.

Ultralytics can send anonymous usage events (no images or video) when its ``sync`` setting is on. This
project disables that for the running process without changing the user's global Ultralytics settings.
Video frames are never sent anywhere by this project.
"""

from __future__ import annotations


def disable_third_party_telemetry() -> None:
    try:
        from ultralytics.utils import events

        events.events.enabled = False
    except Exception:  # noqa: BLE001 - best effort; module layout differs across versions
        pass
