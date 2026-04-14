"""Models tab — per-agent LLM profile cards with apply-to-all."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import HorizontalGroup, VerticalGroup
from textual.widgets import Button, Input, Label, Select, Static
from textual import on

from droidrun.cli.tui.settings.data import (
    AGENT_ROLES,
    PROVIDER_FIELDS,
    PROVIDERS,
    ProfileSettings,
    SettingsData,
)
from droidrun.cli.tui.settings.section import Section


PROVIDER_OPTIONS = [(p, p) for p in PROVIDERS]
API_KEY_SOURCE_OPTIONS = [
    ("auto", "Auto"),
    ("env", "Shell env"),
    ("file", "Saved file"),
]


class _KwargsRow(HorizontalGroup):
    """A single key-value pair row with a remove button."""

    CSS_PATH = "../css/models_tab.tcss"

    def __init__(self, key: str, value: str, row_id: str) -> None:
        super().__init__()
        self._key = key
        self._value = value
        self._row_id = row_id

    def compose(self) -> ComposeResult:
        yield Input(
            value=self._key,
            placeholder="key",
            classes="kwarg-key",
            id=f"kk-{self._row_id}",
        )
        yield Input(value=self._value, placeholder="value", id=f"kv-{self._row_id}")
        yield Button("×", id=f"kr-{self._row_id}")


class _KwargsEditor(VerticalGroup):
    """Editable key-value pair list."""

    CSS_PATH = "../css/models_tab.tcss"

    def __init__(self, kwargs: dict[str, str], role: str) -> None:
        super().__init__()
        self._kwargs = kwargs
        self._role = role
        self._counter = 0

    def compose(self) -> ComposeResult:
        for key, value in self._kwargs.items():
            rid = f"{self._role}-{self._counter}"
            self._counter += 1
            yield _KwargsRow(key, value, rid)
        yield Button("+ add", classes="kwargs-add-btn", id=f"kwargs-add-{self._role}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id == f"kwargs-add-{self._role}":
            event.stop()
            rid = f"{self._role}-{self._counter}"
            self._counter += 1
            self.mount(_KwargsRow("", "", rid), before=event.button)
        elif btn_id.startswith("kr-"):
            event.stop()
            row = event.button.parent
            if row:
                row.remove()

    def collect(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for row in self.query(_KwargsRow):
            key_input = row.query("Input.kwarg-key")
            val_input = row.query("Input:not(.kwarg-key)")
            if key_input and val_input:
                k = key_input.first().value.strip()
                v = val_input.first().value.strip()
                if k:
                    result[k] = v
        return result


class _ProfileCard(Section):
    """Full LLM config card for one agent role."""

    CSS_PATH = "../css/models_tab.tcss"

    def __init__(self, role: str, profile: ProfileSettings) -> None:
        super().__init__(title=role.title())
        self._role = role
        self._profile = profile

    def compose(self) -> ComposeResult:
        if self._role == "manager":
            yield Button("Apply to all", id="apply-all-btn")

        pf = PROVIDER_FIELDS.get(self._profile.provider, {})

        with VerticalGroup(classes="profile-fields"):
            with HorizontalGroup(classes="field-row"):
                yield Label("Provider", classes="field-label")
                yield Select(
                    PROVIDER_OPTIONS,
                    value=self._profile.provider,
                    allow_blank=False,
                    id=f"provider-{self._role}",
                    classes="field-select",
                )

            with HorizontalGroup(classes="field-row"):
                yield Label("Model", classes="field-label")
                yield Input(
                    value=self._profile.model,
                    id=f"model-{self._role}",
                    classes="field-input",
                )

            api_key_cls = "field-row" if pf.get("api_key") else "field-row hidden-field"
            with HorizontalGroup(classes=api_key_cls, id=f"row-apikey-{self._role}"):
                yield Label("API Key", classes="field-label")
                yield Input(
                    value=self._profile.api_key,
                    password=True,
                    id=f"apikey-{self._role}",
                    classes="field-input",
                )

            source_cls = (
                "field-row" if pf.get("api_source") else "field-row hidden-field"
            )
            with HorizontalGroup(classes=source_cls, id=f"row-apikey-source-{self._role}"):
                yield Label("Key Source", classes="field-label")
                yield Select(
                    API_KEY_SOURCE_OPTIONS,
                    value=self._profile.api_key_source,
                    allow_blank=False,
                    id=f"apikey-source-{self._role}",
                    classes="field-select",
                )

            url_cls = "field-row" if pf.get("base_url") else "field-row hidden-field"
            with HorizontalGroup(classes=url_cls, id=f"row-baseurl-{self._role}"):
                yield Label("Base URL", classes="field-label")
                yield Input(
                    value=self._profile.base_url,
                    placeholder="http://localhost:11434",
                    id=f"baseurl-{self._role}",
                    classes="field-input",
                )

            with HorizontalGroup(classes="field-row"):
                yield Label("Temperature", classes="field-label")
                yield Input(
                    value=str(self._profile.temperature),
                    id=f"temp-{self._role}",
                    classes="field-input",
                )

        yield Static("extra parameters", classes="kwargs-label")
        yield _KwargsEditor(self._profile.kwargs, self._role)

    @on(Select.Changed)
    def _on_provider_changed(self, event: Select.Changed) -> None:
        if event.select.id != f"provider-{self._role}":
            return
        provider = str(event.value)
        pf = PROVIDER_FIELDS.get(provider, {})

        api_row = self.query_one(f"#row-apikey-{self._role}")
        source_row = self.query_one(f"#row-apikey-source-{self._role}")
        url_row = self.query_one(f"#row-baseurl-{self._role}")

        if pf.get("api_key"):
            api_row.remove_class("hidden-field")
        else:
            api_row.add_class("hidden-field")

        if pf.get("api_source"):
            source_row.remove_class("hidden-field")
        else:
            source_row.add_class("hidden-field")

        if pf.get("base_url"):
            url_row.remove_class("hidden-field")
        else:
            url_row.add_class("hidden-field")

    def collect(self) -> ProfileSettings:
        provider = str(self.query_one(f"#provider-{self._role}", Select).value)
        model = self.query_one(f"#model-{self._role}", Input).value.strip()
        api_key = self.query_one(f"#apikey-{self._role}", Input).value.strip()
        api_key_source = str(
            self.query_one(f"#apikey-source-{self._role}", Select).value)
        base_url = self.query_one(f"#baseurl-{self._role}", Input).value.strip()
        temp_str = self.query_one(f"#temp-{self._role}", Input).value.strip()
        try:
            temperature = float(temp_str)
        except (ValueError, TypeError):
            temperature = 0.2
        kwargs = self.query_one(_KwargsEditor).collect()
        return ProfileSettings(
            provider=provider,
            model=model,
            temperature=temperature,
            api_key=api_key,
            api_key_source=api_key_source,
            base_url=base_url,
            kwargs=kwargs,
        )


class ModelsTab(VerticalGroup):
    """Content for the Models tab pane — per-agent profile cards."""

    CSS_PATH = "../css/models_tab.tcss"

    def __init__(self, settings: SettingsData) -> None:
        super().__init__()
        self.settings = settings

    def compose(self) -> ComposeResult:
        for role in AGENT_ROLES:
            profile = self.settings.profiles.get(role, ProfileSettings())
            yield _ProfileCard(role, profile)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply-all-btn":
            event.stop()
            self._apply_first_to_all()

    def _apply_first_to_all(self) -> None:
        """Copy the first profile card's values to all other cards."""
        cards = list(self.query(_ProfileCard))
        if len(cards) < 2:
            return
        source = cards[0].collect()
        for card in cards[1:]:
            card.query_one(f"#provider-{card._role}", Select).value = source.provider
            card.query_one(f"#model-{card._role}", Input).value = source.model
            card.query_one(f"#apikey-{card._role}", Input).value = source.api_key
            card.query_one(f"#apikey-source-{card._role}", Select).value = (
                source.api_key_source
            )
            card.query_one(f"#baseurl-{card._role}", Input).value = source.base_url
            card.query_one(f"#temp-{card._role}", Input).value = str(source.temperature)
            # Trigger field visibility update
            pf = PROVIDER_FIELDS.get(source.provider, {})
            api_row = card.query_one(f"#row-apikey-{card._role}")
            source_row = card.query_one(f"#row-apikey-source-{card._role}")
            url_row = card.query_one(f"#row-baseurl-{card._role}")
            if pf.get("api_key"):
                api_row.remove_class("hidden-field")
            else:
                api_row.add_class("hidden-field")
            if pf.get("api_source"):
                source_row.remove_class("hidden-field")
            else:
                source_row.add_class("hidden-field")
            if pf.get("base_url"):
                url_row.remove_class("hidden-field")
            else:
                url_row.add_class("hidden-field")

    def collect(self) -> dict[str, ProfileSettings]:
        result: dict[str, ProfileSettings] = {}
        for card in self.query(_ProfileCard):
            result[card._role] = card.collect()
        return result
