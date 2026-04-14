from __future__ import annotations

from pathlib import Path

import platformdirs

APP_NAME = "droidrun"
USER_CONFIG_DIR = Path(platformdirs.user_config_dir(APP_NAME))
OAUTH_CREDENTIAL_DIR = USER_CONFIG_DIR / "credentials"

# Single credential file for all providers (OAuth tokens + API keys).
AUTH_PROFILES_PATH = OAUTH_CREDENTIAL_DIR / "auth-profiles.json"

# Legacy aliases — all point to the same file now.
OPENAI_OAUTH_CREDENTIAL_PATH = AUTH_PROFILES_PATH
ANTHROPIC_OAUTH_CREDENTIAL_PATH = AUTH_PROFILES_PATH
GEMINI_OAUTH_CREDENTIAL_PATH = AUTH_PROFILES_PATH
