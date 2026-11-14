from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import click
from rich.console import Console
from rich.panel import Panel

from mobilerun.agent.providers.minimax import (
    MINIMAX_CHINA_BASE_URL,
    MINIMAX_GLOBAL_BASE_URL,
)
from mobilerun.agent.providers.registry import (
    VARIANT_ENV_KEY_SLOT,
    model_display_name_for_variant,
    resolve_provider_variant,
)
from mobilerun.agent.providers.setup_service import (
    SetupSelection,
    apply_selection_to_roles,
    auth_mode_choices,
    family_choices,
    variant_models,
)
from mobilerun.cli.configure_prompts import (
    SelectChoice,
    select_prompt,
    text_prompt,
)
from mobilerun.config_manager import ConfigLoader
from mobilerun.config_manager.config_manager import (
    DecisionProfile,
    DelegationConfig,
    LLMProfile,
)
from mobilerun.config_manager.env_keys import (
    load_env_key_sources,
    resolve_env_key,
    save_env_keys,
)

_BACK = "__back__"
_CUSTOM_MINIMAX_BASE_URL = "__custom_minimax_base_url__"

_ALL_CONFIG_ROLES = (
    "manager",
    "executor",
    "fast_agent",
    "app_opener",
    "structured_output",
    "delegate",
)
_DECISION_ROLES = ("fast_agent", "executor")
_AGENT_ROLES = tuple(role for role in _ALL_CONFIG_ROLES if role != "delegate")


@dataclass
class ConfigureWizardCallbacks:
    run_openai_oauth_login: Callable[..., None]
    run_anthropic_oauth_login: Callable[..., None]
    run_gemini_oauth_login: Callable[..., None]
    run_grok_oauth_login: Callable[..., None] | None = None


@dataclass
class ConfigureWizardState:
    family_id: str | None = None
    selected_auth_mode: str | None = None
    selected_model: str | None = None
    selected_api_key: str | None = None
    selected_api_key_source: str | None = None
    selected_base_url: str | None = None
    last_variant_id: str | None = None
    prepared_auth_variant_id: str | None = None
    used_advanced_settings: bool = False


def _with_back_choice(
    choices: list[SelectChoice], *, include_back: bool = True
) -> list[SelectChoice]:
    if not include_back:
        return choices
    return [*choices, SelectChoice(value=_BACK, label="Back")]


def _select_with_back(
    message: str,
    choices: list[SelectChoice],
    *,
    default: str | None = None,
    include_back: bool = True,
) -> str:
    return select_prompt(
        message,
        _with_back_choice(choices, include_back=include_back),
        default=default,
    )


def _print_configure_intro(console: Console) -> None:
    console.print(
        Panel(
            "Choose your provider, auth method, and model.\n"
            "Advanced agent settings are optional and can be changed at the end.",
            title="Mobilerun Configure",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def _print_configure_summary(
    console: Console,
    *,
    family_id: str,
    auth_mode: str,
    provider_label: str,
    variant_id: str,
    model: str,
    used_advanced_settings: bool,
) -> None:
    advanced_line = "Yes" if used_advanced_settings else "No"
    provider_detail = "" if provider_label == "XAI" else f" ({variant_id})"
    model_label = model_display_name_for_variant(family_id, auth_mode, model)
    console.print(
        Panel(
            f"Provider: {provider_label}{provider_detail}\n"
            f"Model: {model_label}\n"
            f"Advanced settings changed: {advanced_line}",
            title="Configuration Saved",
            border_style="green",
            padding=(1, 2),
        )
    )


def _prompt_int(console: Console, message: str, default: int) -> int:
    while True:
        value = text_prompt(message, default=str(default))
        try:
            return int(value)
        except ValueError:
            console.print("[red]Please enter a whole number.[/]")


def _prompt_float(console: Console, message: str, default: float) -> float:
    while True:
        value = text_prompt(message, default=str(default))
        try:
            return float(value)
        except ValueError:
            console.print("[red]Please enter a number.[/]")


def _prompt_model_choice(
    models: list[str],
    *,
    family_id: str,
    auth_mode: str,
    default_model: str,
    allow_back: bool = True,
) -> str:
    if models:
        choice = _select_with_back(
            "Choose model",
            [
                *[
                    SelectChoice(
                        value=item,
                        label=model_display_name_for_variant(
                            family_id, auth_mode, item
                        ),
                    )
                    for item in models
                ],
                SelectChoice(
                    value="enter_model",
                    label="Enter custom model",
                ),
            ],
            default=default_model or None,
            include_back=allow_back,
        )
        if choice in {_BACK, "enter_model"}:
            if choice == _BACK:
                return _BACK
            return text_prompt("Model", default=default_model)
        return choice

    choice = _select_with_back(
        "Choose model",
        [
            SelectChoice(
                value="enter_model",
                label="Enter custom model",
            )
        ],
        default="enter_model",
        include_back=allow_back,
    )
    if choice == _BACK:
        return _BACK
    return text_prompt("Model", default=default_model)


def _prompt_api_key_source(variant: Any) -> str:
    env_slot = VARIANT_ENV_KEY_SLOT.get(variant.id)
    sources = load_env_key_sources().get(env_slot) if env_slot else None
    choices: list[SelectChoice] = []
    if sources and sources.shell:
        choices.append(SelectChoice(value="env", label="Use env key"))
    if sources and sources.saved:
        choices.append(SelectChoice(value="file", label="Use saved key"))
    choices.append(SelectChoice(value="paste", label="Paste new key"))
    return _select_with_back("API key source", choices, default=choices[0].value)


def _prompt_api_key_for_variant(variant: Any) -> tuple[str, str]:
    env_slot = VARIANT_ENV_KEY_SLOT.get(variant.id)
    if not env_slot:
        return text_prompt("API key", secret=True), "file"

    source = _prompt_api_key_source(variant)
    if source == _BACK:
        return "", _BACK
    if source == "env":
        return resolve_env_key(env_slot, "env"), "env"
    if source == "file":
        return resolve_env_key(env_slot, "file"), "file"
    return text_prompt("API key", secret=True), "file"


def _prompt_base_url_for_variant(variant: Any) -> str:
    if variant.id != "MiniMax":
        return text_prompt("Base URL", default=variant.base_url or "", secret=False)

    selected = select_prompt(
        "Choose MiniMax API region",
        [
            SelectChoice(
                value=MINIMAX_GLOBAL_BASE_URL,
                label="Global",
                hint=MINIMAX_GLOBAL_BASE_URL,
            ),
            SelectChoice(
                value=MINIMAX_CHINA_BASE_URL,
                label="Mainland China",
                hint=MINIMAX_CHINA_BASE_URL,
            ),
            SelectChoice(
                value=_CUSTOM_MINIMAX_BASE_URL,
                label="Custom endpoint",
            ),
        ],
        default=MINIMAX_GLOBAL_BASE_URL,
    )
    if selected == _CUSTOM_MINIMAX_BASE_URL:
        return text_prompt("Custom MiniMax base URL", secret=False)
    return selected


# OAuth variants share one auth-profiles.json file but store tokens under
# distinct nested slots. Detection must be slot-aware: the file existing (e.g.
# from another provider, or the deprecated gemini "geminiOauth" slot) does NOT
# mean THIS variant has usable credentials.
_OAUTH_CREDENTIAL_SLOTS = {
    "gemini_oauth_code_assist": "geminiAntigravityOauth",
    "openai_oauth": "openaiOauth",
    "anthropic_oauth": "claudeAiOauth",
    "xai_oauth": "grokOauth",
}


def _oauth_credentials_present(credential_path: str, variant_id: str) -> bool:
    path = Path(credential_path).expanduser()
    if not path.exists():
        return False
    slot = _OAUTH_CREDENTIAL_SLOTS.get(variant_id)
    if not slot:
        return True  # unknown variant: fall back to file presence
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    nested = data.get(slot) if isinstance(data, dict) else None
    if not isinstance(nested, dict):
        return False
    # Token field names vary by provider: gemini access_token/refresh_token,
    # anthropic accessToken/refreshToken, openai access/refresh. Accept any.
    for key, val in nested.items():
        kl = key.lower()
        if val and ("token" in kl or kl in ("access", "refresh")):
            return True
    return False


def _prompt_oauth_credential_action(credential_path: str) -> str:
    return _select_with_back(
        "OAuth credentials found",
        [
            SelectChoice(value="use_existing", label="Use existing login"),
            SelectChoice(value="login_again", label="Log in again"),
        ],
        default="use_existing",
    )


def _apply_model_selection(
    config,
    *,
    family_id: str,
    variant: Any,
    selected_auth_mode: str,
    selected_model: str,
    selected_api_key: str | None,
    selected_api_key_source: str | None,
    selected_base_url: str | None,
    credential_path: str | None,
    roles: tuple[str, ...] | None = None,
) -> None:
    selection = SetupSelection(
        family_id=family_id,
        variant_id=variant.id,
        auth_mode=selected_auth_mode,
        model=selected_model,
        api_key=selected_api_key,
        api_key_source=selected_api_key_source or "auto",
        base_url=selected_base_url,
        credential_path=credential_path,
    )
    if roles is None:
        roles = _roles_following_the_llm(config)
    apply_selection_to_roles(config, selection, roles)


def _roles_following_the_llm(config) -> tuple[str, ...]:
    """All roles, minus a delegate set to a different LLM than the agent."""
    delegate = config.llm_profiles.get("delegate")
    main = config.llm_profiles.get("fast_agent")
    if delegate is not None and main is not None and delegate != main:
        return tuple(role for role in _ALL_CONFIG_ROLES if role != "delegate")
    return _ALL_CONFIG_ROLES


def _prepare_variant_auth(
    *,
    callbacks: ConfigureWizardCallbacks,
    variant: Any,
    credential_path: str | None,
    selected_model: str,
) -> None:
    if variant.id == "openai_oauth" and credential_path:
        callbacks.run_openai_oauth_login(
            credential_path=credential_path, model=selected_model
        )
    elif variant.id == "anthropic_oauth" and credential_path:
        callbacks.run_anthropic_oauth_login(credential_path=credential_path)
    elif variant.id == "gemini_oauth_code_assist" and credential_path:
        callbacks.run_gemini_oauth_login(
            credential_path=credential_path, model=selected_model
        )
    elif variant.id == "xai_oauth" and credential_path:
        if callbacks.run_grok_oauth_login is None:
            raise RuntimeError("XAI OAuth login callback is not configured.")
        callbacks.run_grok_oauth_login(
            credential_path=credential_path,
            model=selected_model,
        )


def _set_profile_max_tokens(profile: Any, value: int) -> None:
    profile.kwargs = dict(profile.kwargs)
    profile.kwargs["max_tokens"] = value


def _set_profile_context_window(profile: Any, value: int) -> None:
    profile.kwargs = dict(profile.kwargs)
    profile.kwargs["context_window"] = value


def _any_ollama_profile(config) -> bool:
    return any(
        getattr(profile, "provider", "") == "Ollama"
        for profile in config.llm_profiles.values()
    )


def _any_minimax_profile(config) -> bool:
    return any(
        getattr(profile, "provider_family", "") == "minimax"
        for profile in config.llm_profiles.values()
    )


def _toggle_label(enabled: bool) -> str:
    """Return a toggle indicator: ON/OFF."""
    return "[ON]" if enabled else "[OFF]"


def _is_vision_enabled(config) -> bool:
    """Check if vision is enabled on any agent."""
    return (
        config.agent.manager.vision
        or config.agent.executor.vision
        or config.agent.fast_agent.vision
    )


def _set_vision_all(config, enabled: bool) -> None:
    """Set vision on all agents at once."""
    config.agent.manager.vision = enabled
    config.agent.executor.vision = enabled
    config.agent.fast_agent.vision = enabled


def _configure_advanced_settings(
    console: Console,
    config,
) -> None:
    default_selection = "vision"
    while True:
        vision_on = _is_vision_enabled(config)
        reasoning_on = config.agent.reasoning

        choices = [
            SelectChoice(
                value="vision",
                label=f"Vision {_toggle_label(vision_on)}",
            ),
            SelectChoice(
                value="reasoning",
                label=f"Reasoning {_toggle_label(reasoning_on)}",
            ),
            SelectChoice(
                value="max_steps",
                label="Maximum steps",
            ),
            SelectChoice(
                value="timeout",
                label=(
                    f"Workflow timeout ({config.agent.timeout}s)"
                    if config.agent.timeout
                    else "Workflow timeout (none)"
                ),
            ),
            SelectChoice(
                value="temperature",
                label="Temperature",
            ),
            SelectChoice(
                value="max_tokens",
                label="Max tokens",
            ),
        ]
        if _any_ollama_profile(config):
            choices.append(
                SelectChoice(
                    value="context_window",
                    label="Context window (Ollama)",
                )
            )
        choices.append(SelectChoice(value="done", label="Done"))

        selected = _select_with_back(
            "Advanced settings",
            choices,
            default=default_selection,
        )

        if selected in {_BACK, "done"}:
            return

        if selected == "vision":
            _set_vision_all(config, not vision_on)
        elif selected == "reasoning":
            config.agent.reasoning = not reasoning_on
        elif selected == "max_steps":
            config.agent.max_steps = _prompt_int(
                console, "Maximum steps", default=config.agent.max_steps
            )
        elif selected == "timeout":
            while True:
                value = _prompt_int(
                    console,
                    "Workflow timeout in seconds (0 = no timeout)",
                    default=config.agent.timeout,
                )
                if value >= 0:
                    break
                console.print("[red]Timeout must be 0 or a positive integer.[/red]")
            config.agent.timeout = value
        elif selected == "temperature":
            default_temp = config.llm_profiles[_ALL_CONFIG_ROLES[0]].temperature
            while True:
                value = _prompt_float(console, "Temperature", default=default_temp)
                if not _any_minimax_profile(config) or 0 < value <= 1:
                    break
                console.print(
                    "[red]MiniMax temperature must be greater than 0 and no "
                    "more than 1.[/red]"
                )
            for role in _ALL_CONFIG_ROLES:
                if role in config.llm_profiles:
                    config.llm_profiles[role].temperature = value
        elif selected == "max_tokens":
            current_value = config.llm_profiles[_ALL_CONFIG_ROLES[0]].kwargs.get(
                "max_tokens", 1024
            )
            try:
                current_default = int(current_value)
            except (TypeError, ValueError):
                current_default = 1024
            value = _prompt_int(console, "Max tokens", default=current_default)
            for role in _ALL_CONFIG_ROLES:
                if role in config.llm_profiles:
                    _set_profile_max_tokens(config.llm_profiles[role], value)
        elif selected == "context_window":
            ollama = [
                profile
                for profile in config.llm_profiles.values()
                if profile.provider == "Ollama"
            ]
            current_value = ollama[0].kwargs.get("context_window", 32768)
            try:
                current_default = int(current_value)
            except (TypeError, ValueError):
                current_default = 32768
            value = _prompt_int(
                console, "Context window (-1 = model max)", default=current_default
            )
            for profile in ollama:
                _set_profile_context_window(profile, value)

        default_selection = selected


def _decision_mode(config) -> str:
    if not any(role in config.decision_profiles for role in _DECISION_ROLES):
        return "off"
    return "delegate" if config.agent.delegation is not None else "alone"


def _llm_delegate_name(config, role: str) -> str | None:
    """Name of the separate LLM profile the current delegation hands steps to."""
    delegation = config.agent.delegation
    if delegation is None:
        return None
    target = delegation.delegate
    if (
        target in config.llm_profiles
        and target not in config.decision_profiles
        and target != (delegation.primary or role)
    ):
        return target
    return None


def _free_delegate_name(config) -> str:
    """A profile name for the delegate LLM that no decision profile shadows."""
    name, n = "delegate", 1
    while name in config.decision_profiles:
        n += 1
        name = f"delegate_{n}"
    return name


def ensure_llm_delegation(config, role: str, profile: Any = None) -> None:
    """Make the decision model hand typing and unsure steps to an LLM.

    *profile* replaces the delegate LLM; without it, a delegation that already
    names a separate LLM profile is kept, and a new one copies the action LLM.
    Profiles that agent roles also use are never overwritten.
    """
    delegation = config.agent.delegation
    target = _llm_delegate_name(config, role)
    if target is not None:
        if profile is None:
            return
        if target not in _AGENT_ROLES:
            config.llm_profiles[target] = profile
            return
    name = _free_delegate_name(config)
    if profile is not None:
        config.llm_profiles[name] = profile
    elif name not in config.llm_profiles:
        source = role if role in config.llm_profiles else "fast_agent"
        if source not in config.llm_profiles:
            source = next(iter(config.llm_profiles), None)
        if source is None:
            raise ValueError(
                "Delegation needs an LLM profile to hand steps to, and the "
                "config has none."
            )
        config.llm_profiles[name] = deepcopy(config.llm_profiles[source])
    if delegation is None:
        config.agent.delegation = DelegationConfig(delegate=name)
        return
    # Keep the primary and threshold; a decision delegate's description no
    # longer fits an LLM.
    config.agent.delegation = replace(
        delegation,
        delegate=name,
        description=(
            None
            if delegation.delegate in config.decision_profiles
            else delegation.description
        ),
    )


def _prompt_typesafe_key_source() -> str:
    """Pick, or paste and save, the TypeSafe key; returns api_key_source."""
    sources = load_env_key_sources()["typesafe"]
    choices: list[SelectChoice] = []
    if sources.shell:
        choices.append(SelectChoice(value="env", label="Use env key"))
    if sources.saved:
        choices.append(SelectChoice(value="file", label="Use saved key"))
    choices.append(SelectChoice(value="paste", label="Paste new key"))
    source = _select_with_back("TypeSafe API key", choices, default=choices[0].value)
    if source != "paste":
        return source
    key = text_prompt("API key", secret=True).strip()
    if not key:
        return _BACK
    save_env_keys({"typesafe": key})
    return "file"


def _profile_label(profile: Any) -> str:
    return f"{profile.provider} · {profile.model}"


def _prompt_delegate_llm(
    config, role: str, choose_llm: Callable[[], Any] | None
) -> Any:
    """Pick the LLM for delegated steps; None keeps the current one."""
    main = config.llm_profiles.get(role) or config.llm_profiles.get("fast_agent")
    current = config.llm_profiles.get(_llm_delegate_name(config, role) or "delegate")
    choices: list[SelectChoice] = []
    if current is not None and (main is None or current != main):
        choices.append(
            SelectChoice(value="keep", label=f"Keep {_profile_label(current)}")
        )
    if main is not None:
        choices.append(
            SelectChoice(
                value="main", label=f"Same as the agent ({_profile_label(main)})"
            )
        )
    if choose_llm is not None:
        choices.append(SelectChoice(value="other", label="Another provider / model"))
    if not choices:
        return None
    answer = _select_with_back(
        "LLM for delegated steps", choices, default=choices[0].value
    )
    if answer == "keep":
        # A delegate shared with an agent role is copied, since that role may
        # become Jev.
        name = _llm_delegate_name(config, role)
        return deepcopy(current) if name is None or name in _AGENT_ROLES else None
    if answer == "main":
        return deepcopy(main)
    if answer == "other":
        return choose_llm() or _BACK
    return _BACK


def _configure_decision_model(
    console: Console, config, choose_llm: Callable[[], Any] | None = None
) -> None:
    """Let TypeSafe's Jev pick actions, handing steps to an LLM by default.

    *choose_llm* runs the provider / model flow and returns the chosen
    profile, or None when the user backs out.
    """
    current = _decision_mode(config)
    mode = _select_with_back(
        "Decision model",
        [
            SelectChoice(
                value="delegate",
                label="TypeSafe Jev picks actions; an LLM types and takes unsure steps",
            ),
            SelectChoice(
                value="alone", label="TypeSafe Jev picks actions (cannot type)"
            ),
            SelectChoice(value="off", label="Off: the LLM picks every action"),
        ],
        default="delegate" if current == "off" else current,
    )
    if mode == _BACK:
        return
    if mode == "off":
        for role in _DECISION_ROLES:
            config.decision_profiles.pop(role, None)
        if current != "off":
            config.agent.delegation = None
        console.print("Decision model: off")
        return
    source = _prompt_typesafe_key_source()
    if source == _BACK:
        return
    role = "executor" if config.agent.reasoning else "fast_agent"
    delegate = None
    if mode == "delegate":
        delegate = _prompt_delegate_llm(config, role, choose_llm)
        if delegate is _BACK:
            return
    for decision_role in _DECISION_ROLES:
        existing = config.decision_profiles.get(decision_role)
        config.decision_profiles[decision_role] = (
            replace(existing, api_key_source=source)
            if existing is not None
            else DecisionProfile(api_key_source=source)
        )
    if mode == "alone":
        config.agent.delegation = None
        console.print("Decision model: TypeSafe Jev")
        return
    try:
        ensure_llm_delegation(config, role, delegate)
    except ValueError as error:
        config.agent.delegation = None
        console.print(f"[yellow]{error} Jev will run alone.[/]")
        return
    target = _llm_delegate_name(config, role) or "delegate"
    console.print(
        "Decision model: TypeSafe Jev, handing steps to "
        f"{_profile_label(config.llm_profiles[target])}"
    )


def _configure_provider_model(
    console: Console,
    config,
    callbacks: ConfigureWizardCallbacks,
    state: ConfigureWizardState,
    families,
    family_labels: dict[str, str],
    *,
    provider_is_fixed: bool,
    auth_mode_is_fixed: bool,
    model_is_fixed: bool,
    api_key: str | None,
    base_url: str | None,
    roles: tuple[str, ...] | None = None,
) -> bool:
    """Run the provider → auth → model → credentials flow.

    Returns True when the flow completes and the selection has been applied
    to *config*.  Returns False when the user backs out entirely.
    """
    while True:
        # --- Provider family ---
        if state.family_id is None:
            state.family_id = _select_with_back(
                "Choose provider",
                [
                    SelectChoice(value=family.id, label=family.display_name)
                    for family in families
                ],
                default="gemini",
                include_back=True,
            )
            if state.family_id == _BACK:
                state.family_id = None
                return False
        console.print(f"Selected provider family: {family_labels[state.family_id]}")

        # --- Auth mode ---
        modes = auth_mode_choices(state.family_id)
        if auth_mode_is_fixed:
            state.selected_auth_mode = click.Choice(
                list(modes), case_sensitive=False
            ).convert(state.selected_auth_mode or "", None, None)
        else:
            while True:
                if len(modes) == 1:
                    state.selected_auth_mode = modes[0]
                    break
                state.selected_auth_mode = _select_with_back(
                    "Choose auth mode",
                    [
                        SelectChoice(value=mode, label=mode.replace("_", " "))
                        for mode in modes
                    ],
                    default=modes[0],
                )
                if state.selected_auth_mode == _BACK:
                    state.family_id = None
                    break
                break
            if state.selected_auth_mode is None:
                continue
            if state.selected_auth_mode == _BACK:
                state.family_id = None
                state.selected_auth_mode = None
                continue

        # --- Model ---
        models = list(variant_models(state.family_id, state.selected_auth_mode))
        variant = resolve_provider_variant(state.family_id, state.selected_auth_mode)
        default_model = models[0] if models else (variant.default_model or "")

        if not model_is_fixed:
            while True:
                state.selected_model = _prompt_model_choice(
                    models,
                    family_id=state.family_id,
                    auth_mode=state.selected_auth_mode,
                    default_model=default_model,
                )
                if state.selected_model == _BACK:
                    state.selected_model = None
                    if provider_is_fixed and (auth_mode_is_fixed or len(modes) == 1):
                        return False
                    if auth_mode_is_fixed or len(modes) == 1:
                        if not provider_is_fixed:
                            state.family_id = None
                        if not auth_mode_is_fixed:
                            state.selected_auth_mode = None
                    else:
                        state.selected_auth_mode = None
                    break
                break
            if state.selected_model is None:
                continue

        # --- Credentials ---
        credential_path: str | None = variant.credential_path

        if variant.id != state.last_variant_id:
            if api_key is None:
                state.selected_api_key = None
                state.selected_api_key_source = None
            if base_url is None:
                state.selected_base_url = None
            state.last_variant_id = variant.id
            state.prepared_auth_variant_id = None

        non_interactive = provider_is_fixed and model_is_fixed

        if variant.requires_api_key and not state.selected_api_key:
            if non_interactive:
                # Auto-resolve key from saved credentials
                env_slot = VARIANT_ENV_KEY_SLOT.get(variant.id)
                if env_slot:
                    key = resolve_env_key(env_slot, "auto")
                    if key:
                        state.selected_api_key = key
                        state.selected_api_key_source = "auto"
                    else:
                        raise click.ClickException(
                            f"No API key found for {variant.id}. "
                            f"Pass --api-key or save one first."
                        )
                else:
                    raise click.ClickException(
                        f"No API key provided for {variant.id}. Pass --api-key."
                    )
            else:
                selected_key, selected_source = _prompt_api_key_for_variant(variant)
                if selected_key == _BACK:
                    if model_is_fixed:
                        return False
                    state.selected_model = None
                    continue
                state.selected_api_key = selected_key
                state.selected_api_key_source = selected_source
        if variant.requires_base_url and not state.selected_base_url:
            if non_interactive and variant.base_url:
                state.selected_base_url = variant.base_url
            else:
                state.selected_base_url = _prompt_base_url_for_variant(variant)
        if (
            credential_path
            and variant.auth_mode == "oauth"
            and state.prepared_auth_variant_id != variant.id
            and _oauth_credentials_present(credential_path, variant.id)
        ):
            if non_interactive:
                state.prepared_auth_variant_id = variant.id
            else:
                oauth_action = _prompt_oauth_credential_action(credential_path)
                if oauth_action == _BACK:
                    if model_is_fixed:
                        return False
                    state.selected_model = None
                    continue
                if oauth_action == "use_existing":
                    state.prepared_auth_variant_id = variant.id

        # --- Apply ---
        if credential_path and state.prepared_auth_variant_id != variant.id:
            _prepare_variant_auth(
                callbacks=callbacks,
                variant=variant,
                credential_path=credential_path,
                selected_model=state.selected_model,
            )
            state.prepared_auth_variant_id = variant.id

        _apply_model_selection(
            config,
            family_id=state.family_id,
            variant=variant,
            selected_auth_mode=state.selected_auth_mode,
            selected_model=state.selected_model,
            selected_api_key=state.selected_api_key,
            selected_api_key_source=state.selected_api_key_source,
            selected_base_url=state.selected_base_url,
            credential_path=credential_path,
            roles=roles,
        )
        return True


def run_configure_wizard(
    console: Console,
    callbacks: ConfigureWizardCallbacks,
    *,
    provider: str | None,
    auth_mode: str | None,
    model: str | None,
    api_key: str | None,
    base_url: str | None,
) -> None:
    config = ConfigLoader.load()
    _print_configure_intro(console)

    families = family_choices()
    family_ids = [family.id for family in families]
    family_labels = {family.id: family.display_name for family in families}
    state = ConfigureWizardState(
        selected_api_key=api_key,
        selected_api_key_source="file" if api_key else None,
        selected_base_url=base_url,
    )

    provider_is_fixed = provider is not None
    auth_mode_is_fixed = auth_mode is not None
    model_is_fixed = model is not None

    if provider_is_fixed:
        state.family_id = click.Choice(family_ids, case_sensitive=False).convert(
            provider, None, None
        )
    if auth_mode_is_fixed:
        state.selected_auth_mode = auth_mode
    if model_is_fixed:
        state.selected_model = model

    # Enter a partially fixed provider flow only when another flag establishes
    # where that flow should resume. A provider-only invocation starts at the
    # top-level menu, matching the fully interactive wizard.
    provider_configured = False
    if provider_is_fixed and (auth_mode_is_fixed or model_is_fixed):
        provider_configured = _configure_provider_model(
            console,
            config,
            callbacks,
            state,
            families,
            family_labels,
            provider_is_fixed=provider_is_fixed,
            auth_mode_is_fixed=auth_mode_is_fixed,
            model_is_fixed=model_is_fixed,
            api_key=api_key,
            base_url=base_url,
        )
        if provider_configured:
            ConfigLoader.save(config)
            _print_configure_summary(
                console,
                family_id=state.family_id or "",
                auth_mode=state.selected_auth_mode or "",
                provider_label=family_labels[state.family_id],
                variant_id=state.last_variant_id or state.family_id,
                model=state.selected_model or "",
                used_advanced_settings=False,
            )
            return

    # --- Top-level menu ---
    while True:
        action = select_prompt(
            "Configure",
            [
                SelectChoice(value="provider_model", label="Provider / Model"),
                SelectChoice(value="decision", label="Decision model"),
                SelectChoice(value="advanced", label="Advanced settings"),
                SelectChoice(value="finish", label="Finish"),
            ],
            default="finish" if provider_configured else "provider_model",
        )

        if action == "provider_model":
            # Reset provider state so the user can re-pick
            state.family_id = None
            state.selected_auth_mode = None
            state.selected_model = None
            state.selected_api_key = api_key
            state.selected_api_key_source = "file" if api_key else None
            state.selected_base_url = base_url
            state.last_variant_id = None
            state.prepared_auth_variant_id = None

            completed = _configure_provider_model(
                console,
                config,
                callbacks,
                state,
                families,
                family_labels,
                provider_is_fixed=False,
                auth_mode_is_fixed=False,
                model_is_fixed=False,
                api_key=api_key,
                base_url=base_url,
            )
            if completed:
                provider_configured = True

        elif action == "decision":

            def choose_delegate_llm():
                # Run the provider flow on a copy so backing out changes nothing.
                scratch = deepcopy(config)
                scratch.llm_profiles["delegate"] = LLMProfile()
                if not _configure_provider_model(
                    console,
                    scratch,
                    callbacks,
                    ConfigureWizardState(),
                    families,
                    family_labels,
                    provider_is_fixed=False,
                    auth_mode_is_fixed=False,
                    model_is_fixed=False,
                    api_key=None,
                    base_url=None,
                    roles=("delegate",),
                ):
                    return None
                config.agent.streaming = scratch.agent.streaming
                return scratch.llm_profiles["delegate"]

            _configure_decision_model(console, config, choose_delegate_llm)

        elif action == "advanced":
            state.used_advanced_settings = True
            _configure_advanced_settings(console, config)

        elif action == "finish":
            ConfigLoader.save(config)
            if provider_configured and state.family_id:
                _print_configure_summary(
                    console,
                    family_id=state.family_id,
                    auth_mode=state.selected_auth_mode or "",
                    provider_label=family_labels[state.family_id],
                    variant_id=state.last_variant_id or state.family_id,
                    model=state.selected_model or "",
                    used_advanced_settings=state.used_advanced_settings,
                )
            else:
                console.print("[green]Configuration saved.[/]")
            return
