"""Contracts for relocated core helpers and legacy compatibility."""

import asyncio
import importlib


def test_prompt_and_path_helpers_live_in_utils():
    prompts = importlib.import_module("mobilerun.utils.prompts")
    paths = importlib.import_module("mobilerun.utils.paths")
    assert (
        prompts.PromptLoader.render_template("Hello {{ name }}", {"name": "world"})
        == "Hello world"
    )
    assert paths.PathResolver.resolve(".").exists()
    assert callable(prompts.PromptLoader.load_prompt)


def test_async_prompt_loader_renders_file(tmp_path):
    from mobilerun.utils.prompts import PromptLoader

    template = tmp_path / "prompt.jinja2"
    template.write_text("Hello {{ name }}", encoding="utf-8")
    assert (
        asyncio.run(PromptLoader.load_prompt(str(template), {"name": "Ada"}))
        == "Hello Ada"
    )
