import base64
import io
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from mobilerun.sandbox.packaging import package
from tests.support.sandbox_playground import extract


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
