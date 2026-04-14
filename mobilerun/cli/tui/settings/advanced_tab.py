"""Advanced tab — TCP, trajectory, tracing, timing."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import HorizontalGroup, VerticalGroup
from textual.widgets import Input, Label, Select
from textual import on

from droidrun.cli.tui.settings.data import SettingsData
from droidrun.cli.tui.settings.section import BoolToggle, Section

TRACING_PROVIDERS = [
    ("Phoenix", "phoenix"),
    ("Langfuse", "langfuse"),
]


class AdvancedTab(VerticalGroup):
    """Content for the Advanced tab pane."""

    CSS_PATH = "../css/advanced_tab.tcss"

    def __init__(self, settings: SettingsData) -> None:
        super().__init__()
        self.settings = settings

    def compose(self) -> ComposeResult:
        with Section("Connection"):
            with HorizontalGroup(classes="field-row"):
                yield Label("Use TCP", classes="field-label")
                yield BoolToggle(value=self.settings.use_tcp, id="use-tcp")

        with Section("Logging"):
            with HorizontalGroup(classes="field-row"):
                yield Label("Debug", classes="field-label")
                yield BoolToggle(value=self.settings.debug, id="debug-logging")

            with HorizontalGroup(classes="field-row"):
                yield Label("Trajectory", classes="field-label")
                yield BoolToggle(
                    value=self.settings.save_trajectory, id="save-trajectory"
                )

            with HorizontalGroup(classes="field-row"):
                yield Label("GIFs", classes="field-label")
                yield BoolToggle(
                    value=self.settings.trajectory_gifs, id="trajectory-gifs"
                )

        with Section("Tracing"):
            with HorizontalGroup(classes="field-row"):
                yield Label("Enabled", classes="field-label")
                yield BoolToggle(
                    value=self.settings.tracing_enabled, id="tracing-enabled"
                )

            with HorizontalGroup(classes="field-row"):
                yield Label("Provider", classes="field-label")
                yield Select(
                    TRACING_PROVIDERS,
                    value=self.settings.tracing_provider,
                    allow_blank=False,
                    id="tracing-provider",
                    classes="field-select",
                )

            langfuse_cls = (
                "field-row"
                if self.settings.tracing_provider == "langfuse"
                else "field-row hidden-field"
            )

            with HorizontalGroup(classes=langfuse_cls, id="row-langfuse-host"):
                yield Label("Host", classes="field-label")
                yield Input(
                    value=self.settings.langfuse_host,
                    placeholder="https://cloud.langfuse.com",
                    id="langfuse-host",
                    classes="field-input",
                )

            with HorizontalGroup(classes=langfuse_cls, id="row-langfuse-pk"):
                yield Label("Public Key", classes="field-label")
                yield Input(
                    value=self.settings.langfuse_public_key,
                    id="langfuse-pk",
                    classes="field-input",
                )

            with HorizontalGroup(classes=langfuse_cls, id="row-langfuse-sk"):
                yield Label("Secret Key", classes="field-label")
                yield Input(
                    value=self.settings.langfuse_secret_key,
                    password=True,
                    id="langfuse-sk",
                    classes="field-input",
                )

            with HorizontalGroup(classes=langfuse_cls, id="row-langfuse-screenshots"):
                yield Label("Screenshots", classes="field-label")
                yield BoolToggle(
                    value=self.settings.langfuse_screenshots, id="langfuse-screenshots"
                )

        with Section("Timing"):
            with HorizontalGroup(classes="field-row"):
                yield Label("Action Sleep", classes="field-label")
                yield Input(
                    value=str(self.settings.after_sleep_action),
                    placeholder="1.0",
                    id="after-sleep-action",
                    classes="field-input",
                )

            with HorizontalGroup(classes="field-row"):
                yield Label("UI Stable Wait", classes="field-label")
                yield Input(
                    value=str(self.settings.wait_for_stable_ui),
                    placeholder="0.3",
                    id="wait-for-stable-ui",
                    classes="field-input",
                )

    @on(Select.Changed, "#tracing-provider")
    def _on_provider_changed(self, event: Select.Changed) -> None:
        is_langfuse = str(event.value) == "langfuse"
        for row_id in (
            "row-langfuse-host",
            "row-langfuse-pk",
            "row-langfuse-sk",
            "row-langfuse-screenshots",
        ):
            row = self.query_one(f"#{row_id}")
            if is_langfuse:
                row.remove_class("hidden-field")
            else:
                row.add_class("hidden-field")

    def collect(self) -> dict:
        """Collect current advanced settings."""
        after_sleep_str = self.query_one("#after-sleep-action", Input).value.strip()
        wait_stable_str = self.query_one("#wait-for-stable-ui", Input).value.strip()
        try:
            after_sleep = float(after_sleep_str)
        except (ValueError, TypeError):
            after_sleep = 1.0
        try:
            wait_stable = float(wait_stable_str)
        except (ValueError, TypeError):
            wait_stable = 0.3

        return {
            "use_tcp": self.query_one("#use-tcp", BoolToggle).value,
            "debug": self.query_one("#debug-logging", BoolToggle).value,
            "save_trajectory": self.query_one("#save-trajectory", BoolToggle).value,
            "trajectory_gifs": self.query_one("#trajectory-gifs", BoolToggle).value,
            "tracing_enabled": self.query_one("#tracing-enabled", BoolToggle).value,
            "tracing_provider": self.query_one("#tracing-provider", Select).value,
            "langfuse_host": self.query_one("#langfuse-host", Input).value.strip(),
            "langfuse_public_key": self.query_one("#langfuse-pk", Input).value.strip(),
            "langfuse_secret_key": self.query_one("#langfuse-sk", Input).value.strip(),
            "langfuse_screenshots": self.query_one(
                "#langfuse-screenshots", BoolToggle
            ).value,
            "after_sleep_action": after_sleep,
            "wait_for_stable_ui": wait_stable,
        }
