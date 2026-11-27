import asyncio
import json
import logging

import pytest

from mobilerun.core.credentials import DictCredentials
from mobilerun.core.devices import AndroidDevice, Device, RecordingDevice, StealthDevice
from mobilerun.core.devices.components.wrapper import DeviceWrapper
from mobilerun.core.extensions import Extension, action, device_action
from mobilerun.core.runs import recording_policy, recording_scope
from mobilerun.core.state.base import StateProvider
from mobilerun.core.state.ui_state import UIState
from mobilerun.utils.redaction import secret_scope
from tests.support.devices import MemoryDevice


class Phone(MemoryDevice):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.waits = []
        self.dumps = 0
        self.screenshots = 0

    async def tap(self, x, y):
        self.calls.append((x, y))
        if x < 0:
            raise ValueError("backend detail must not persist")

    async def _input_text_raw(self, text, clear=False, stealth=False, wpm=0):
        self.calls.append(text)
        return False

    async def wait(self, seconds):
        self.waits.append(seconds)

    async def screenshot(self, hide_overlay=True):
        self.screenshots += 1
        return b"fake png"


class Provider(StateProvider):
    async def get_state(self):
        self._device.dumps += 1
        return UIState(
            [{"resourceId": "button", "text": "free-form screen text"}],
            "formatted text",
            "focused text",
            {"packageName": "app", "activityName": f"View{len(self._device.calls)}"},
            10,
            10,
        )


def rows(recording):
    return [
        json.loads(line)
        for line in (recording.run_dir / "steps.jsonl").read_text().splitlines()
    ]


@pytest.mark.parametrize("stealth_outer", [False, True])
def test_stealth_context_suppresses_actions_in_both_wrapper_orders(
    tmp_path, stealth_outer
):
    class Probe(Extension):
        @action()
        async def inspect(self) -> None:
            """Read context through an older extension implementation."""

        async def context(self):
            await self.inspect()
            return ()

    async def run():
        phone = Phone()
        recorder = RecordingDevice(
            phone if stealth_outer else StealthDevice(phone),
            runs_dir=tmp_path,
            capture="actions",
        )
        facade = StealthDevice(recorder) if stealth_outer else recorder
        await Probe().bind(recorder)
        await DictCredentials({"login": "secret"}).bind(recorder)
        assert await facade.extensions["extension"].context() == ()
        assert (await facade.context())[0].kind == "credentials"
        assert recorder.run_dir is None

    asyncio.run(run())


def test_stealth_undecorated_helper_cannot_record_its_nested_action(tmp_path):
    class Probe(Extension):
        @action()
        async def ready(self) -> bool:
            """Return readiness."""
            return True

        async def helper(self):
            return await self.ready()

    async def run():
        recorder = RecordingDevice(Phone(), runs_dir=tmp_path, capture="actions")
        await Probe(name="probe").bind(recorder)
        assert await StealthDevice(recorder).probe.helper()
        assert recorder.run_dir is None
        assert await recorder.probe.helper()
        assert rows(recorder)[0]["args"]["name"] == "probe.ready"

    asyncio.run(run())


def test_registered_secret_is_masked_at_dispatch_and_omitted_from_macro(
    tmp_path, caplog, monkeypatch
):
    from mobilerun.macro import build_macro, load_macro

    class AcceptingPhone(Phone):
        async def _input_text_raw(self, text, clear=False, stealth=False, wpm=0):
            logging.getLogger(__name__).warning("typed %s", text)
            self.calls.append(text)
            return True

    async def run():
        phone = AcceptingPhone()
        recording = RecordingDevice(
            phone, runs_dir=tmp_path / "runs", capture="actions"
        )
        phone.register_sensitive_values(iter(["s3cret"]))

        async def tool():
            await DeviceWrapper(recording).input_text("s3cret")

        await tool()
        await recording.input_text("hello")
        await recording.input_text("s3cretary")
        assert phone.calls == ["s3cret", "hello", "s3cretary"]
        assert [row["args"]["text"] for row in rows(recording)] == [
            "***",
            "hello",
            "s3cretary",
        ]
        assert caplog.messages == ["typed ***", "typed hello", "typed s3cretary"]
        path = build_macro(recording.run_dir / "steps.jsonl", "secret-free")
        assert [step.args["text"] for step in load_macro(path)] == [
            "hello",
            "s3cretary",
        ]

    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(tmp_path / "macros"))
    asyncio.run(run())


def test_stealth_lazy_helpers_keep_veto_during_iteration_and_close(tmp_path):
    class Probe(Extension):
        @action()
        async def ready(self) -> bool:
            """Return readiness."""
            return True

        async def lazy(self):
            try:
                yield await self.ready()
            finally:
                await self.ready()

        def lazy_sync(self):
            try:
                yield self.device.start_run()
            finally:
                self.device.start_run()

    async def run():
        recorder = RecordingDevice(Phone(), runs_dir=tmp_path, capture="actions")
        await Probe(name="probe").bind(recorder)
        facade = StealthDevice(recorder)
        lazy = facade.probe.lazy()
        assert await anext(lazy)
        assert recording_policy()[0]  # The caller is outside the generator scope.
        assert recorder.run_dir is None
        await lazy.aclose()
        assert recorder.run_dir is None
        sync = facade.probe.lazy_sync()
        assert next(sync) is None
        sync.close()
        assert recorder.run_dir is None

    asyncio.run(run())


def test_known_otp_keeps_live_elements_and_regenerates_sanitized_text():
    from mobilerun.core.state.snapshot import DeviceSnapshot

    class Mail(Extension):
        @action()
        async def wait_for_code(self) -> str:
            """Read a fake verification code."""
            return "482913"

    class Screen(Provider):
        async def get_state(self):
            state = await super().get_state()
            state.elements = [
                {
                    "index": 1,
                    "resourceId": "otp",
                    "text": "482-913",
                    "bounds": "0,0,10,10",
                },
                {
                    "index": 2,
                    "resourceId": "submit",
                    "className": "Button",
                    "text": "Submit",
                    "bounds": "0,0,10,10",
                },
            ]
            state.formatted_text = state.focused_text = "482-913"
            return state

    async def run():
        phone = Phone()
        recorder = RecordingDevice(
            phone, state_provider=Screen(phone), capture="actions"
        )
        await Mail(name="mail").bind(phone)
        await phone.mail.wait_for_code()
        snapshot = await DeviceSnapshot.observe(recorder)
        assert snapshot.ui.get_element(2)["text"] == "Submit"
        assert snapshot.ui.get_element_coords(2) == (5, 5)
        assert snapshot.ui.get_element(1)["text"] == "***"
        assert '2. Button: "submit", "Submit"' in snapshot.ui.formatted_text
        assert "482-913" not in snapshot.ui.formatted_text

    asyncio.run(run())


def test_raw_android_prefilled_password_is_safe_on_disk_and_targetable_live(tmp_path):
    from mobilerun.core.state.snapshot import DeviceSnapshot

    class Android(AndroidDevice):
        async def _get_ui_tree(self):
            return {
                "a11y_tree": {
                    "isPassword": True,
                    "isFocused": False,
                    "resourceId": "login_password",
                    "className": "android.widget.EditText",
                    "text": "raw-password-canary",
                    "boundsInScreen": {
                        "left": 10,
                        "top": 10,
                        "right": 70,
                        "bottom": 30,
                    },
                    "children": [],
                },
                "phone_state": {
                    "packageName": "example.app",
                    "focusedElement": None,
                    "isEditable": False,
                },
                "device_context": {"screen_bounds": {"width": 100, "height": 100}},
            }

        async def tap(self, x, y):
            pass

        async def wait(self, seconds):
            pass

        async def screenshot(self, hide_overlay=True):
            return b"unchanged PNG"

    async def run():
        device = Android("fake")
        recorder = RecordingDevice(device, runs_dir=tmp_path)
        await recorder.tap(10, 10)
        for files in rows(recorder)[0]["screens"].values():
            assert (
                "raw-password-canary"
                not in (recorder.run_dir / files["json"]).read_text()
            )
            assert (recorder.run_dir / files["png"]).read_bytes() == b"unchanged PNG"
        live = await DeviceSnapshot.observe(device)
        assert live.ui.get_element(1)["text"] == "***"
        assert live.ui.get_element_coords(1) == (40, 20)
        assert '1. EditText: "login_password", "***"' in live.ui.formatted_text
        assert "raw-password-canary" not in live.ui.formatted_text

    asyncio.run(run())


def test_full_observation_keeps_vault_context_without_dispatching_actions(tmp_path):
    class Probe(Extension):
        @action()
        async def ready(self) -> bool:
            """Return a non-sensitive result."""
            return True

    async def run():
        phone = Phone()
        recorder = RecordingDevice(
            phone, state_provider=Provider(phone), runs_dir=tmp_path
        )
        await DictCredentials({"login": "private-vault-value"}).bind(recorder)
        await Probe().bind(recorder)
        await recorder.tap(1, 2)
        assert await recorder.extension.ready()
        assert [row["action"] for row in rows(recorder)] == ["tap", "extension_call"]
        after = rows(recorder)[-1]["screens"]["after"]["json"]
        data = (recorder.run_dir / after).read_text()
        assert json.loads(data)["device_context"][0]["kind"] == "credentials"
        assert "private-vault-value" not in data

    asyncio.run(run())


def test_tainted_fingerprints_are_omitted_after_secret_scope_and_on_cache_hits(
    tmp_path, monkeypatch
):
    from mobilerun.macro import build_macro

    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(tmp_path / "macros"))
    secret = "dynamic-secret-canary"

    class DynamicProvider(Provider):
        async def get_state(self):
            state = await super().get_state()
            if secret in self._device.calls:
                state.elements[0]["resourceId"] = secret
            return state

    async def run():
        phone = Phone()
        recorder = RecordingDevice(
            phone, state_provider=DynamicProvider(phone), runs_dir=tmp_path / "runs"
        )
        with secret_scope(secret):
            await recorder.input_text(secret)
        assert await recorder.capture_fingerprint() is None
        await recorder.tap(1, 2)
        assert rows(recorder)[-1]["before"] is rows(recorder)[-1]["after"] is None
        path = recorder.run_dir / "steps.jsonl"
        assert secret not in path.read_text()
        assert secret not in build_macro(path, "no-leak").read_text()

    asyncio.run(run())


@pytest.mark.parametrize(
    "secure",
    [
        {"isPassword": True},
        {"password": "true"},
        {"secure": True},
        {"inputType": "textPassword"},
        {"inputType": 129},
        {"className": "XCUIElementTypeSecureTextField"},
    ],
)
def test_prefilled_secure_values_are_omitted_from_json_not_png(tmp_path, secure):
    class PasswordProvider(Provider):
        async def get_state(self):
            state = await super().get_state()
            state.elements.append(
                {
                    "children": [
                        {
                            **secure,
                            "value": "prefilled-canary",
                            "text": "prefilled-canary",
                        }
                    ]
                }
            )
            state.phone_state["focusedElement"] = {"text": "prefilled-canary"}
            state.formatted_text = state.focused_text = "prefilled-canary"
            return state

    async def run():
        phone = Phone()
        recorder = RecordingDevice(
            phone, state_provider=PasswordProvider(phone), runs_dir=tmp_path
        )
        await recorder.tap(1, 2)
        for files in rows(recorder)[0]["screens"].values():
            data = (recorder.run_dir / files["json"]).read_text()
            assert "prefilled-canary" not in data
            assert "free-form screen text" in data
            assert (recorder.run_dir / files["png"]).read_bytes() == b"fake png"

    asyncio.run(run())


def test_finish_detaches_observer_without_closing_device(tmp_path):
    class Probe(Extension):
        @action()
        async def ready(self) -> bool:
            """Return readiness."""
            return True

    async def run():
        phone = Phone()
        recorder = RecordingDevice(phone, runs_dir=tmp_path, capture="actions")
        probe = await Probe().bind(phone)
        assert await recorder.extension.ready()
        recorder.finish("succeeded")
        recorder.finish("succeeded")
        assert not phone._extension_state.observers
        assert await probe.ready()
        await recorder.tap(2, 3)
        assert phone.calls == [(2, 3)]
        assert [row["action"] for row in rows(recorder)] == ["extension_call"]
        await recorder.close()
        assert not phone._extension_state.observers

    asyncio.run(run())


def test_finish_while_action_is_in_flight_does_not_write_or_break_action(tmp_path):
    async def run():
        entered, resume = asyncio.Event(), asyncio.Event()

        class BlockingPhone(Phone):
            async def tap(self, x, y):
                entered.set()
                await resume.wait()
                await super().tap(x, y)

        phone = BlockingPhone()
        recorder = RecordingDevice(phone, runs_dir=tmp_path, capture="actions")
        pending = asyncio.create_task(recorder.tap(1, 2))
        await entered.wait()
        recorder.finish("succeeded")
        resume.set()
        await pending
        assert phone.calls == [(1, 2)]
        assert rows(recorder) == []

    asyncio.run(run())


def test_verification_code_is_not_written_to_run_or_macro(tmp_path, monkeypatch):
    from mobilerun.macro import build_macro, load_macro

    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(tmp_path / "macros"))

    class Mail(Extension):
        role = "mailbox"

        @action()
        async def wait_for_code(self) -> str:
            """Read a fake verification code."""
            return "731284"

        @action()
        async def echo(self, value: dict[str, list[str | int]]) -> bool:
            """Accept a nested input."""
            return True

    class AcceptingPhone(Phone):
        async def _input_text_raw(self, text, clear=False, stealth=False, wpm=0):
            self.calls.append(text)
            return True

    async def run():
        phone = AcceptingPhone()
        recorder = RecordingDevice(
            phone, state_provider=Provider(phone), runs_dir=tmp_path
        )
        await Mail(name="email").bind(phone)
        code = await recorder.mailbox.wait_for_code()
        assert await recorder.input_text(code)
        assert await recorder.mailbox.echo({"values": [code, int(code)]})
        await recorder.tap(1, 2)
        path = recorder.run_dir / "steps.jsonl"
        assert phone.calls[0] == code
        assert code not in path.read_text()
        assert [
            row["args"]["text"]
            for row in rows(recorder)
            if row["action"] == "input_text"
        ] == ["***"]
        assert phone.screenshots > 0
        assert all(
            code not in p.read_text() for p in path.parent.glob("screens/*.json")
        )
        macro = build_macro(path, "verification")
        assert code not in macro.read_text()
        assert [step.action for step in load_macro(macro)] == ["extension_call", "tap"]

    asyncio.run(run())


def test_message_code_formats_are_masked_in_logs_and_action_rows(tmp_path, caplog):
    class Number(Extension):
        role = "phone_number"

        @action()
        async def messages(self) -> list[dict[str, str]]:
            """Return an inbound SMS."""
            return [{"body": "Verification code: 482913"}]

    async def run():
        phone = Phone()
        recording = RecordingDevice(phone, runs_dir=tmp_path, capture="actions")
        await Number(name="sms").bind(phone)
        redact = StealthDevice(recording).known_codes_redactor()
        assert redact("use 482 913") == "use 482 913"
        assert await asyncio.create_task(phone.sms.messages()) == [
            {"body": "Verification code: 482913"}
        ]
        logging.getLogger(__name__).warning(
            "code: %s", "482-913", extra={"pin": "482 913"}
        )
        await recording.input_text("482-913")
        assert phone.calls == ["482-913"]
        assert rows(recording)[0]["args"]["text"] == "***"
        assert "code: ***" in caplog.text
        assert caplog.records[-1].pin == "***"
        assert redact("use 482 913") == "use ***"
        assert Phone().known_codes_redactor()("482913") == "482913"

    asyncio.run(run())


@pytest.mark.parametrize("catalog", [False, True])
def test_stealth_extension_veto_reaches_the_bound_action_body(tmp_path, catalog):
    class Probe(Extension):
        @device_action
        async def try_start(self, device: Device) -> bool:
            """Try to start a run on the original bound facade."""
            return device.recording_device().start_run() is not None

    async def run():
        recorder = RecordingDevice(Phone(), runs_dir=tmp_path, capture="actions")
        await Probe(name="probe").bind(recorder)
        probe = StealthDevice(recorder).extensions["probe"]
        result = (
            await probe.actions.call("probe.try_start", {})
            if catalog
            else await probe.try_start()
        )
        assert result is False
        assert not list(tmp_path.glob("*/run.json"))
        assert await recorder.probe.try_start()

    asyncio.run(run())


def test_step_collection_inherits_into_new_tasks_but_excludes_existing_tasks(tmp_path):
    async def run():
        recorder = RecordingDevice(Phone(), runs_dir=tmp_path, capture="actions")
        proceed = asyncio.Event()

        async def sibling():
            await proceed.wait()
            await recorder.tap(7, 8)

        pending = asyncio.create_task(sibling())
        with recorder.collect_steps() as collected:
            await recorder.tap(1, 2)
            await asyncio.create_task(recorder.tap(3, 4))
            proceed.set()
            await pending
            with recorder.collect_steps() as nested:
                await asyncio.create_task(recorder.tap(5, 6))
        assert collected == [1, 2, 4]
        assert nested == [4]
        assert recorder.step_index == 4

    asyncio.run(run())


def test_close_cancels_active_manifest(tmp_path):
    async def run():
        recorder = RecordingDevice(Phone(), runs_dir=tmp_path, capture="actions")
        await recorder.tap(1, 2)
        await recorder.close()
        assert (
            json.loads((recorder.run_dir / "run.json").read_text())["status"]
            == "cancelled"
        )

    asyncio.run(run())


def test_failed_finish_detaches_and_context_preserves_body_exception(
    tmp_path, monkeypatch
):
    from mobilerun.core.runs import RunWriter

    def fail(*args, **kwargs):
        raise OSError("manifest failure")

    monkeypatch.setattr(RunWriter, "finish", fail)

    async def run():
        recorder = RecordingDevice(Phone(), runs_dir=tmp_path, capture="actions")
        with pytest.raises(KeyError, match="body failed"):
            async with recorder:
                await recorder.tap(1, 2)
                raise KeyError("body failed")
        assert not recorder._extension_state.observers
        other = RecordingDevice(Phone(), runs_dir=tmp_path, capture="actions")
        other.start_run()
        with pytest.raises(OSError, match="manifest failure"):
            other.finish("succeeded")
        assert not other._extension_state.observers
        await other.tap(2, 3)
        assert rows(other) == []

    asyncio.run(run())


def test_records_outcomes_and_reuses_current_and_after_samples(tmp_path):
    phone = Phone()
    recorder = RecordingDevice(phone, state_provider=Provider(phone), runs_dir=tmp_path)

    async def run():
        await recorder.state.refresh()
        assert recorder.run_dir is None
        with recording_scope(by="agent"):
            await recorder.tap(1, 2)
        with pytest.raises(ValueError, match="backend detail"):
            await recorder.tap(-1, 2)
        assert await recorder.input_text("plain", clear=True, wpm=12) is False
        recorder.finish("failed", failed_step=2)

    asyncio.run(run())
    first, second, third = rows(recorder)
    assert [(r["i"], r["ok"], r["error"], r["by"]) for r in (first, second, third)] == [
        (1, True, None, "agent"),
        (2, False, "ValueError", "script"),
        (3, False, "returned_false", "script"),
    ]
    assert first["before"] == {
        "app": "app",
        "activity": "View0",
        "elements": ["id:button"],
    }
    assert first["after"] == second["before"]
    assert second["after"] == third["before"]
    assert third["args"] == {
        "text": "plain",
        "clear": True,
        "stealth": False,
        "wpm": 12,
    }
    assert phone.dumps == 4
    assert phone.waits == [0.25] * 3
    assert recorder.step_index == 3
    manifest = json.loads((recorder.run_dir / "run.json").read_text())
    assert (manifest["status"], manifest["failed_step"]) == ("failed", 2)


@pytest.mark.parametrize("stealth_outer", [True, False])
def test_stealth_vetoes_both_wrapper_orders(tmp_path, stealth_outer):
    phone = Phone()
    runs_dir = tmp_path / "runs"
    if stealth_outer:
        recorder = RecordingDevice(phone, runs_dir=runs_dir)
        facade = StealthDevice(recorder)
    else:
        facade = recorder = RecordingDevice(StealthDevice(phone), runs_dir=runs_dir)
    assert facade.recording_device() is recorder
    assert not facade.recording_allowed()
    if not stealth_outer:
        assert recorder.start_run() is None
    asyncio.run(facade.tap(1, 2))
    assert phone.calls == [(1, 2)]
    assert not runs_dir.exists()


def test_lookup_reuses_recorder_and_scope_veto_cannot_be_lifted(tmp_path):
    phone = Phone()
    assert phone.recording_device() is None
    recorder = RecordingDevice(phone, runs_dir=tmp_path, capture="actions")
    facade = DeviceWrapper(RecordingDevice(recorder, runs_dir=tmp_path / "unused"))
    assert facade.recording_device() is recorder
    assert facade.recording_allowed()

    async def run():
        with recording_scope(enabled=False), recording_scope(enabled=True, by="agent"):
            assert recorder.start_run() is None
            await facade.tap(1, 2)
        assert recorder.step_index == 0
        await facade.tap(3, 4)

    asyncio.run(run())
    assert phone.calls == [(1, 2), (3, 4)]
    assert [r["args"] for r in rows(recorder)] == [{"x": 3, "y": 4}]


def test_forced_fingerprint_refresh_and_suppressed_dispatch_invalidate_cache(tmp_path):
    phone = Phone()
    recorder = RecordingDevice(phone, state_provider=Provider(phone), runs_dir=tmp_path)

    async def run():
        assert (await recorder.capture_fingerprint()).activity == "View0"
        await phone.tap(1, 2)
        assert (await recorder.capture_fingerprint()).activity == "View0"
        assert (await recorder.capture_fingerprint(fresh=True)).activity == "View1"
        with recording_scope(enabled=False):
            await recorder.tap(3, 4)
        assert recorder.run_dir is None
        assert (await recorder.capture_fingerprint()).activity == "View2"

    asyncio.run(run())
    assert phone.dumps == 3


def test_actions_capture_skips_observation_and_replays(tmp_path, monkeypatch):
    from mobilerun.macro import MacroPlayer, build_macro, load_macro

    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(tmp_path / "macros"))
    phone = Phone()
    recorder = RecordingDevice(
        phone, state_provider=Provider(phone), runs_dir=tmp_path, capture="actions"
    )

    async def run():
        await recorder.tap(1, 2)
        with secret_scope("sensitive-canary"):
            assert not await recorder.input_text("sensitive-canary")
        await recorder.tap(3, 4)
        assert phone.dumps == phone.screenshots == 0
        assert not phone.waits
        macro = load_macro(
            build_macro(recorder.run_dir / "steps.jsonl", "actions-only")
        )
        replay = Phone()
        await MacroPlayer(replay, delay_between_actions=0).replay_macro(macro)
        assert replay.calls == [(1, 2), (3, 4)]
        assert await recorder.capture_fingerprint(fresh=True) is None

    asyncio.run(run())
    public, sensitive, _ = rows(recorder)
    assert public["before"] is public["after"] is None
    assert phone.screenshots == 0
    assert sensitive["args"]["text"] == "***"
    assert sensitive["screens"] is None
    assert not (recorder.run_dir / "screens").exists()
    assert "sensitive-canary" not in (recorder.run_dir / "steps.jsonl").read_text()


def test_snapshots_and_action_timing_reuse_previous_after(tmp_path):
    from datetime import datetime

    phone = Phone()
    recorder = RecordingDevice(phone, state_provider=Provider(phone), runs_dir=tmp_path)

    async def run():
        await recorder.tap(1, 2)
        await recorder.tap(3, 4)
        with secret_scope("sensitive-canary"):
            await recorder.input_text("sensitive-canary")

    asyncio.run(run())
    first, second, sensitive = rows(recorder)
    assert first["screens"]["before"]["json"] == "screens/1.before.json"
    assert first["screens"]["after"]["json"] == "screens/1.after.json"
    assert second["screens"]["before"] == first["screens"]["after"]
    for phase in ("before", "after"):
        assert (
            recorder.run_dir / sensitive["screens"][phase]["png"]
        ).read_bytes() == b"fake png"
    assert (
        json.loads(
            (recorder.run_dir / sensitive["screens"]["after"]["json"]).read_text()
        )["ui"]["focused_text"]
        == "***"
    )
    snapshot = json.loads(
        (recorder.run_dir / first["screens"]["before"]["json"]).read_text()
    )
    assert snapshot["ui"]["formatted_text"] == "formatted text"
    assert snapshot["device_context"] == []
    assert phone.dumps == 4
    assert "png" in first["screens"]["before"]
    assert datetime.fromisoformat(first["started_at"]) <= datetime.fromisoformat(
        first["finished_at"]
    )


def test_writer_failure_does_not_redispatch(tmp_path):
    phone = Phone()
    recorder = RecordingDevice(phone, runs_dir=tmp_path, capture="actions")

    async def run():
        run_dir = recorder.start_run()
        (run_dir / "steps.jsonl").unlink()
        with pytest.raises(OSError):
            await recorder.tap(1, 2)
        with pytest.raises(RuntimeError, match="closed for recording"):
            await recorder.tap(3, 4)

    asyncio.run(run())
    assert phone.calls == [(1, 2)]
    assert recorder.step_index == 0


def test_direct_secret_dispatch_records_only_a_masked_literal(tmp_path):
    class EditableProvider(Provider):
        async def get_state(self):
            state = await super().get_state()
            state.phone_state.update(
                focusedElement={"className": "EditText"}, isEditable=True
            )
            return state

    phone = Phone()
    recorder = RecordingDevice(
        phone,
        state_provider=EditableProvider(phone),
        runs_dir=tmp_path,
        capture="actions",
    )
    assert asyncio.run(recorder._input_secret("sensitive-canary")) is False
    (row,) = rows(recorder)
    assert (row["action"], row["args"]["text"], row["ok"]) == (
        "input_text",
        "***",
        False,
    )
