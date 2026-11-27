"""Offline playground contracts: endpoint calls, extraction, snapshots, and deployment output."""

import ast
import base64
import io
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from click.testing import CliRunner
from mobilerun_functions import RunInfo
from PIL import Image

from mobilerun.sandbox.cli.group import sandbox
from mobilerun.sandbox.documents import build_document, equivalent_document
from mobilerun.sandbox.models import (
    Artifact,
    CloudDeclaration,
    DeploymentReceipt,
    Endpoint,
)
from mobilerun.sandbox.packaging import package
from tests.support import sandbox_playground
from tests.support.sandbox_playground import extract, ping
from tests.support.sandbox_playground import main as snapshot
from tests.support.sandbox_playground import settings as script

_CONNECTION_OVERRIDES = re.compile(
    r"https?://\S+"
    r"|\bBearer\s+(?!<)[\w.-]+"
    r"|\b(?:MOBILERUN_)?(?:API_KEY|BASE_URL)\s*[:=]\s*['\"]?\S+"
    r"|\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


@pytest.fixture
def caller(monkeypatch):
    from tests.support.sandbox_playground import call_endpoint

    monkeypatch.setattr(call_endpoint, "time", SimpleNamespace(monotonic=lambda: 0.0))
    monkeypatch.setenv("MOBILERUN_API_KEY", "fake-test-credential")
    return call_endpoint


def test_nested_long_base64_is_redacted(caller):
    encoded = base64.b64encode(b"abc" * 1000).decode()
    assert caller.redact({"nested": [encoded]})["nested"][0] == "<base64 jpeg, 3 KB>"


def test_pretty_print_and_save_screenshot(caller, tmp_path, capsys):
    image = b"jpeg" * 400
    screenshot = base64.b64encode(image).decode()
    cloud = MagicMock()
    run = cloud["demo"].fire.return_value
    run.invocation_id = "inv"
    run.id = "exec"
    run.result.return_value = {"message": "ok", "screenshot": screenshot}

    assert caller.main("demo", output_dir=tmp_path, endpoints=cloud) == 0
    assert (tmp_path / "demo.last.jpg").read_bytes() == image
    out = capsys.readouterr().out
    assert "<base64 jpeg, 2 KB>" in out
    assert screenshot not in out
    assert "fired invocation inv" in out
    assert "execution exec running" in out
    cloud["demo"].fire.assert_called_once_with()


def test_missing_key(caller, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("MOBILERUN_API_KEY")
    cloud = MagicMock()
    assert caller.main("demo", output_dir=tmp_path, endpoints=cloud) == 1
    assert "run `source ~/.agent-env` first" in capsys.readouterr().err
    cloud.__getitem__.assert_not_called()


@pytest.mark.parametrize("name", ["../escape", "/absolute", r"..\escape", "..", ""])
def test_rejects_path_traversal_before_firing(caller, name, tmp_path, capsys):
    cloud = MagicMock()
    output_dir = tmp_path / "images"
    assert caller.main(name, output_dir=output_dir, endpoints=cloud) == 1
    assert "Invalid endpoint name" in capsys.readouterr().err
    assert not output_dir.exists()
    assert not (tmp_path / "escape.last.jpg").exists()
    cloud.__getitem__.assert_not_called()


def test_refuses_existing_symlink(caller, tmp_path, capsys):
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"unchanged")
    output_dir = tmp_path / "images"
    output_dir.mkdir()
    (output_dir / "demo.last.jpg").symlink_to(outside)
    cloud = MagicMock()
    assert caller.main("demo", output_dir=output_dir, endpoints=cloud) == 1
    assert "Invalid screenshot path" in capsys.readouterr().err
    assert outside.read_bytes() == b"unchanged"
    cloud.__getitem__.assert_not_called()


@pytest.mark.parametrize(
    "unsafe",
    [
        "https://example.test/private",
        "Bearer fake-credential",
        "API_KEY='fake-credential'",
        "01234567-89ab-cdef-0123-456789abcdef",
    ],
)
def test_generic_connection_hygiene_patterns(unsafe):
    assert _CONNECTION_OVERRIDES.search(unsafe)


def test_playground_fixtures_have_no_credentials_urls_or_device_ids():
    root = Path(sandbox_playground.__file__).parent
    forbidden = _CONNECTION_OVERRIDES
    for path in (*sorted(root.glob("*.py")), *sorted(root.glob("*.json"))):
        source = path.read_text(encoding="utf-8")
        assert not forbidden.search(path.name), path.name
        assert not forbidden.search(source), path.name
        if path.suffix == ".py":
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.keyword) and node.arg in {
                    "device",
                    "api_key",
                    "base_url",
                }:
                    assert (
                        isinstance(node.value, ast.Constant)
                        and node.value.value is None
                    ), path.name
                if isinstance(node, ast.Dict):
                    for key in node.keys:
                        if isinstance(key, ast.Constant):
                            assert key.value not in {
                                "MOBILERUN_API_KEY",
                                "MOBILERUN_BASE_URL",
                                "MOBILERUN_CLOUD_BASE_URL",
                            }, path.name
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    assert not forbidden.search(node.value), path.name


class NewsDevice:
    def __init__(self, image_path):
        self.screens = json.loads(
            Path(extract.__file__).with_name("extract_screens.json").read_text()
        )
        self.image_path = image_path
        self.page = "feed_one"
        self.stack = []
        self.actions = []
        self.failed_story = None
        self.stalled = False

    def start_app(self, app):
        assert app == "com.example.news"
        self.actions.append(("open", app))

    def ui(self):
        return SimpleNamespace(
            nodes=self.screens[self.stack[-1] if self.stack else self.page]
        )

    def tap_node(self, node, *, humanize):
        assert not humanize
        assert node in self.ui().nodes
        row = node["resource_id"].removeprefix("story:")
        self.actions.append(("tap", row))
        if row == self.failed_story:
            raise RuntimeError("unavailable")
        self.stack.append("interview" if row == "show_transcript" else row)

    def key(self, key):
        assert key == "back"
        self.stack.pop()

    def swipe(self, x1, y1, x2, y2, *, humanize):
        assert x1 == x2 and y1 > y2 and not humanize and not self.stack
        self.actions.append(("scroll", self.page))
        if not self.stalled:
            self.page = "feed_two"

    def screenshot(self):
        assert self.stack[-1] in {"garden", "library", "mural"}
        return self.image_path


@pytest.fixture
def run_news(tmp_path):
    shot = tmp_path / "synthetic.png"
    Image.new("RGB", (1200, 800), "teal").save(shot)
    device = NewsDevice(shot)
    ctx = SimpleNamespace(device=lambda: device, partial=Mock())

    def run(**options):
        return extract.main(extract.Input(**options), ctx)

    return run, device, ctx


def test_traverses_feed_pages_and_extracts_unique_typed_stories(run_news):
    run, device, ctx = run_news
    result = run()
    assert [(story.title, story.source) for story in result.stories] == [
        ("A garden opens", "Juniper Gazette"),
        ("Books after dark", "Pebble Press"),
        ("A brighter wall", "Juniper Gazette"),
    ]
    assert result.stories[0].published_at == datetime(
        2026, 1, 2, 9, 30, tzinfo=timezone.utc
    )
    assert device.actions[0] == ("open", "com.example.news")
    assert ("scroll", "feed_one") in device.actions
    assert device.actions.count(("tap", "library")) == 1
    assert not device.stack
    ctx.partial.assert_not_called()
    assert package(Path(extract.__file__)).declaration.name == "news-headlines"


def test_reads_ordered_visible_transcript_and_returns_to_feed(run_news):
    run, device, _ = run_news
    stories = run().stories
    assert stories[0].transcript == "Host: What will you grow?\nGuest: Mint and basil."
    assert stories[1].transcript is None
    assert device.actions.count(("tap", "show_transcript")) == 1
    assert not device.stack


def test_rejects_detail_for_a_different_story(run_news):
    run, device, ctx = run_news
    device.screens["garden"][0]["text"] = "An unrelated headline"
    assert [story.title for story in run().stories] == [
        "Books after dark",
        "A brighter wall",
    ]
    assert "detail did not match" in ctx.partial.call_args.args[0]
    assert not device.stack


def test_failed_screen_reports_partial_without_losing_other_stories(run_news):
    run, device, ctx = run_news
    device.failed_story = "library"
    assert [story.title for story in run().stories] == [
        "A garden opens",
        "A brighter wall",
    ]
    ctx.partial.assert_called_once()
    assert "could not open" in ctx.partial.call_args.args[0]


def test_images_are_valid_resized_jpegs_within_shared_budget(run_news):
    run, _, ctx = run_news
    budget = 12000
    shots = [story.image for story in run(image_budget=budget).stories]
    assert all(shots)
    assert sum(map(len, shots)) <= budget
    for shot in shots:
        with Image.open(io.BytesIO(base64.b64decode(shot, validate=True))) as image:
            assert image.format == "JPEG"
            assert image.size == (640, 427)
    ctx.partial.assert_not_called()


def test_uncompressible_image_does_not_discard_story_text(run_news):
    run, device, ctx = run_news
    pixels = random.Random(7).randbytes(640 * 480 * 3)
    Image.frombytes("RGB", (640, 480), pixels).save(device.image_path)
    stories = run(image_budget=1000).stories
    assert len(stories) == 3
    assert all(story.image is None for story in stories)
    assert "Mint and basil" in stories[0].transcript
    ctx.partial.assert_called_once()
    assert "image budget" in ctx.partial.call_args.args[0]


def test_repeated_feed_stops_without_retapping_rows(run_news):
    run, device, ctx = run_news
    device.stalled = True
    assert len(run(max_pages=5).stories) == 2
    assert device.actions.count(("scroll", "feed_one")) == 1
    assert device.actions.count(("tap", "garden")) == 1
    ctx.partial.assert_not_called()


def test_collects_device_snapshot_and_opens_requested_app(tmp_path):
    shot = tmp_path / "screen.png"
    shot.write_bytes(b"png")
    device = Mock(device_id="fake-device")
    device.screenshot.return_value = str(shot)
    device.ui.return_value = SimpleNamespace(
        nodes=[{"text": "Hello"}, {"text": ""}, {"text": "World"}]
    )
    log = Mock()
    ctx = SimpleNamespace(
        device=lambda: device, log=log, run=RunInfo(device_id=device.device_id)
    )

    with patch("time.monotonic", return_value=0):
        result = snapshot.main(snapshot.Input(app_package="example.app"), ctx)

    assert result.device_id == "fake-device"
    assert result.screenshot_path == str(shot)
    assert result.ui_node_count == 3
    assert result.top_texts == ["Hello", "World"]
    device.start_app.assert_called_once_with("example.app")
    logged = " ".join(call.args[0] for call in log.info.call_args_list)
    assert "Opening app example.app" in logged
    assert "3 nodes, 2 with text" in logged
    log.warning.assert_not_called()


@pytest.mark.parametrize("name", [None, "someone"])
def test_ping_is_constant_for_all_names(name):
    assert ping.main(ping.Input(name=name), None).model_dump() == {
        "message": "pong",
        "version": 1,
    }


def test_device_free_document_preserves_payload_schema():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    artifact = Artifact(CloudDeclaration("playground-ping"), "", "", schema, None)
    doc = build_document(artifact, "function-id", 1, device=None)
    assert "devices" not in doc
    assert doc["on"]["custom"]["inputs"] == schema
    assert doc["steps"][0]["with"]["input"] == "{{event.payload}}"


def test_device_free_document_does_not_match_device_bound_flow():
    artifact = Artifact(CloudDeclaration("playground-ping"), "", "", {}, None)
    doc = build_document(artifact, "function-id", 1, device=None)
    assert not equivalent_document({**doc, "devices": ["fake-device"]}, doc)


@pytest.fixture
def endpoint_contract(monkeypatch):
    endpoint = Endpoint(
        "https://example.test/v1/triggers/trigger-id/fire", "trigger-id"
    )
    receipt = DeploymentReceipt(
        "playground-ping",
        "function-id",
        1,
        "created",
        "flow-id",
        "trigger-id",
        (),
        "digest",
        endpoint,
        (),
    )
    deploy_file = AsyncMock(return_value=receipt)
    from importlib import import_module

    monkeypatch.setattr(
        import_module("mobilerun.sandbox.deployment"), "deploy_file", deploy_file
    )
    result = CliRunner().invoke(sandbox, ["deploy", str(Path(ping.__file__)), "--yes"])
    assert result.exit_code == 0, result.output
    deploy_file.assert_awaited_once()
    return result.output


def test_contract_reports_sdk_trigger_and_flow(endpoint_contract):
    assert "/triggers/trigger-id/fire" in endpoint_contract
    assert "flow-id" in endpoint_contract


def test_contract_reports_auth_payload_and_invocation_poll(endpoint_contract):
    expected = ("Bearer <YOUR_API_KEY>", '"payload": {}', "invocationId", "/executions")
    missing = [text for text in expected if text not in endpoint_contract]
    assert not missing, f"CLI deploy output is missing: {missing}"


def run_settings(ctx):
    with patch.multiple("time", monotonic=lambda: 0.0, sleep=Mock()):
        return script.main(script.Input(wait_s=0), ctx)


@pytest.fixture
def run_device(tmp_path):
    def run(app="com.android.settings"):
        path = tmp_path / "screen.png"
        Image.new("RGB", (1200, 1800), "blue").save(path)
        device = Mock()
        device.screenshot.return_value = str(path)
        device.current_app_id.return_value = app
        device.ui.return_value = SimpleNamespace(
            nodes=[{"text": "Settings"}, {"text": ""}, {"text": "Network"}]
        )
        ctx = SimpleNamespace(device=lambda: device, log=Mock(), partial=Mock())
        return run_settings(ctx), device, ctx

    return run


def test_success_and_home_cleanup(run_device):
    result, device, ctx = run_device()
    assert result.settings_open
    assert result.texts == ["Settings", "Network"]
    assert device.key.call_count == 2
    assert device.key.call_args_list[0].args == ("home",)
    assert device.key.call_args_list[-1].args == ("home",)
    device.start_app.assert_called_once_with("com.android.settings")
    with Image.open(io.BytesIO(base64.b64decode(result.screenshot))) as image:
        assert max(image.size) <= 1000
    assert len(result.screenshot) <= 240000
    ctx.partial.assert_not_called()


def test_wrong_app(run_device):
    result, _, _ = run_device(app="other")
    assert not result.settings_open


def test_budget_exceeded_marks_partial(run_device):
    with patch("base64.b64encode", return_value=b"x" * 300000):
        result, _, ctx = run_device()
    assert result.screenshot is None
    ctx.partial.assert_called_once()


def test_endpoint_document_binds_device():
    artifact = Artifact(CloudDeclaration("x"), "", "", {}, None)
    document = build_document(artifact, "fn", 1, device="fake-device")
    assert document["devices"] == ["fake-device"]
    assert "custom" in document["on"]


def test_home_after_failure():
    device = Mock()
    device.start_app.side_effect = RuntimeError("failed")
    ctx = SimpleNamespace(device=lambda: device, log=Mock())
    with pytest.raises(RuntimeError, match="failed"):
        run_settings(ctx)
    assert device.key.call_count == 2
    assert device.key.call_args.args == ("home",)
