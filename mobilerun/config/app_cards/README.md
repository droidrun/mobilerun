# App Cards

App cards provide app-specific guidance to Mobilerun agents. They help agents understand how to operate specific apps more effectively.

## How It Works

1. **Mapping File**: `app_cards.json` maps Android package names and iOS bundle ids to markdown files
2. **App Card Files**: Markdown files containing app-specific guidance
3. **Automatic Loading**: Mobilerun automatically loads the appropriate app card based on the current package name
4. **Prompt Injection**: App cards are injected into agent prompts when available

## File Structure

```
config/app_cards/
├── app_cards.json       # Package name → file mapping
├── gmail.md             # Gmail app card
├── chrome.md            # Chrome app card
└── social/              # Organize in subdirectories if needed
    └── whatsapp.md
```

## Creating App Cards

### 1. Add entry to app_cards.json

```json
{
  "com.google.android.gm": "gmail.md",
  "com.android.chrome": "chrome.md",
  "com.whatsapp": "social/whatsapp.md"
}
```

### 2. Create the markdown file

Create a `.md` file with guidance about the app:

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

Card paths in `app_cards.json` are resolved like this:

1. **Relative to the folder that contains `app_cards.json`** (most common):
   ```json
   {"com.google.android.gm": "gmail.md", "com.whatsapp": "social/whatsapp.md"}
   ```

2. **Absolute path**:
   ```json
   {"com.google.android.gm": "/usr/share/mobilerun/cards/gmail.md"}
   ```

The `app_cards_dir` setting itself is looked up in the working directory first, then in the package directory.

## Different Cards per Platform

Some apps use the same id on Android and iOS. Map the id to one path per platform; `default` is used for any platform without its own entry:

```json
{
  "com.zhiliaoapp.musically": {
    "android": "android/tiktok.md",
    "ios": "ios/tiktok.md"
  }
}
```

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
