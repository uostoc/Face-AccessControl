from __future__ import annotations


class AlertService:
    def __init__(self, enable_led: bool = True, enable_buzzer: bool = False) -> None:
        self.enable_led = enable_led
        self.enable_buzzer = enable_buzzer

    def notify(self, event_type: str) -> None:
        if event_type == "stranger" and self.enable_led:
            print("[ALERT] LED alert: stranger event")
        if event_type == "stranger" and self.enable_buzzer:
            print("[ALERT] Buzzer alert: stranger event")

