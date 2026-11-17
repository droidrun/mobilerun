from conftest import Secrets, run

from mobilerun.macro.replay import MacroPlayer
from mobilerun.macro.state import MACRO_SCHEMA_VERSION, normalize_ui_state

EMAIL = "jane@example.com"


def _node(res, cls, text):
    return {"resourceId": f"com.shop:id/{res}", "className": cls, "text": text}


def _login_screen(email=""):
    return {
        "a11y_tree": [
            _node("title", "android.widget.TextView", "Sign in"),
            _node("email", "android.widget.EditText", email),
            _node("password", "android.widget.EditText", ""),
            _node("remember", "android.widget.CheckBox", "Remember me"),
            _node("forgot", "android.widget.TextView", "Forgot password?"),
            _node("submit", "android.widget.Button", "Sign in"),
            _node("signup", "android.widget.Button", "Create account"),
        ],
        "phone_state": {"package": "com.shop", "activity": ".Login"},
    }


def _locked_screen():
    return {
        "a11y_tree": [
            _node("message", "android.widget.TextView", "Account locked"),
            _node("ok", "android.widget.Button", "OK"),
        ],
        "phone_state": {"package": "com.shop", "activity": ".Locked"},
    }


class LoginDriver:
    platform = "android"

    def __init__(self, after_input=_login_screen):
        self.after_input = after_input
        self.tree = _login_screen()
        self.taps = []

    async def connect(self):
        return None

    async def get_ui_tree(self):
        return self.tree

    async def tap(self, x, y):
        self.taps.append((x, y))

    async def input_text(self, text, clear=False):
        self.tree = self.after_input(text)
        return True


def _macro(email_after_typing):
    def step(screen, **action):
        return {"pre_state": normalize_ui_state(screen), **action}

    typed = _login_screen(email_after_typing)
    return {
        "macro_schema_version": MACRO_SCHEMA_VERSION,
        "actions": [
            step(_login_screen(), action_type="tap", x=10, y=20),
            step(_login_screen(), action_type="type_secret", secret_id="email"),
            step(typed, action_type="tap", x=10, y=40),
            step(typed, action_type="tap", x=10, y=60),
        ],
    }


def _player(driver, value=EMAIL):
    player = MacroPlayer(
        delay_between_actions=0, state_timeout=0, state_poll_interval=0
    )
    player.driver = driver
    player.credential_manager = Secrets([], {"email": value})
    return player


def test_recorded_state_with_the_clear_text_secret_replays_to_the_end():
    player = _player(LoginDriver())

    assert run(player.replay_macro(_macro(EMAIL))) is True

    assert player.driver.taps == [(10, 20), (10, 40), (10, 60)]


def test_recorded_state_with_the_secret_placeholder_replays_to_the_end():
    player = _player(LoginDriver())

    assert run(player.replay_macro(_macro("[secret:email]"))) is True

    assert player.driver.taps == [(10, 20), (10, 40), (10, 60)]


def test_secret_value_found_inside_a_placeholder_does_not_stop_the_replay():
    player = _player(LoginDriver(), value="secret")

    assert run(player.replay_macro(_macro("[secret:email]"))) is True

    assert player.driver.taps == [(10, 20), (10, 40), (10, 60)]


def test_different_screen_after_the_secret_still_stops_the_replay(caplog):
    player = _player(LoginDriver(after_input=lambda text: _locked_screen()))

    with caplog.at_level("DEBUG"):
        assert run(player.replay_macro(_macro(EMAIL))) is False

    assert player.driver.taps == [(10, 20)]
    assert player.last_divergence["step"] == 3
    assert EMAIL not in caplog.text
    assert EMAIL not in str(player.last_divergence)
