# App Cards

App cards provide app-specific guidance to Mobilerun agents. They help agents understand how to operate specific apps more effectively.

## How It Works

1. **One folder per platform**: `android/` and `ios/`, each with its own `app_cards.json`
2. **Mapping File**: each `app_cards.json` maps an Android package name or iOS bundle id to a markdown file in that folder
3. **App Card Files**: Markdown files containing app-specific guidance
4. **Automatic Loading**: Mobilerun loads the card for the current app from the folder of the device's platform
5. **Prompt Injection**: App cards are injected into agent prompts when available

## File Structure

```
config/app_cards/
├── android/
│   ├── app_cards.json   # Android package name → file mapping
│   ├── gmail.md
│   └── social/          # Organize in subdirectories if needed
│       └── whatsapp.md
└── ios/
    ├── app_cards.json   # iOS bundle id → file mapping
    └── gmail.md
```

An app with the same id on both platforms (for example TikTok, `com.zhiliaoapp.musically`) gets one entry in each folder, each pointing to its own card.

## Creating App Cards

### 1. Add an entry to the platform's app_cards.json

`android/app_cards.json`:

```json
{
  "com.google.android.gm": "gmail.md",
  "com.whatsapp": "social/whatsapp.md"
}
```

### 2. Create the markdown file

Create a `.md` file with guidance about the app in the same folder:

```markdown
# App Name Guide

## Navigation
- How to navigate the app
- Key screens and menus

## Common Actions
- List of common tasks
- How to perform them

## Tips
- App-specific tips
- Known issues or quirks
```

## Path Resolution

- Card paths are relative to the platform folder (`android/` or `ios/`), or absolute.
- An `app_cards.json` directly in `app_cards_dir` is not read; Mobilerun logs a warning and ignores it.
- A relative `app_cards_dir` is used from the working directory if it contains app cards there, otherwise from the package directory.

## Finding Package Names

To find an app's package name:

1. **Using ADB**:
   ```bash
   adb shell pm list packages | grep keyword
   ```

2. **From device**:
   - Open the app
   - Run Mobilerun with debug mode to see the current package name in logs

3. **Common apps**:
   - Gmail: `com.google.android.gm`
   - Chrome: `com.android.chrome`
   - WhatsApp: `com.whatsapp`
   - Instagram: `com.instagram.android`
   - Facebook: `com.facebook.katana`

## Configuration

Enable/disable app cards in `config.yaml`:

```yaml
agent:
  app_cards:
    enabled: true
    app_cards_dir: config/app_cards
```

## Best Practices

1. **Be Concise**: Keep app cards focused and actionable
2. **Use Examples**: Show concrete examples of common tasks
3. **Update Regularly**: Keep app cards current with app updates
4. **Test**: Verify that guidance actually helps agents
5. **Organize**: Use subdirectories for related apps (e.g., social/, banking/)

## Programmatic Usage

```python
import asyncio

from mobilerun.app_cards.providers import (
    CompositeAppCardProvider,
    LocalAppCardProvider,
    ServerAppCardProvider,
)
from mobilerun.config_manager import ConfigLoader


async def main():
    config = ConfigLoader.load()

    # Check if enabled and load a local app card.
    if not config.agent.app_cards.enabled:
        return

    provider = LocalAppCardProvider(app_cards_dir=config.agent.app_cards.app_cards_dir)
    app_card = await provider.load_app_card(
        package_name="com.google.android.gm",
        instruction="Summarize unread email",
        platform="android",
    )
    print(app_card)

    # Server-backed and server-with-local-fallback modes use:
    server_provider = ServerAppCardProvider(server_url="https://example.com")
    composite_provider = CompositeAppCardProvider(
        server_url="https://example.com",
        app_cards_dir=config.agent.app_cards.app_cards_dir,
    )

    # Clear cache and get cache statistics.
    provider.clear_cache()
    stats = provider.get_cache_stats()
    print(f"Cached entries: {stats['content_entries']}")


asyncio.run(main())
```
