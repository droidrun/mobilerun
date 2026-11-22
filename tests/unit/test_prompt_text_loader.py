from mobilerun.config.prompts import load_prompt_text


def test_load_prompt_text_reads_bundled_file_and_matches_previous_guidance():
    assert load_prompt_text("state/screenshot_only_guidance.md") == (
        "Prefer click_at on the center of visible text or controls, especially in "
        "dense lists, adjacent rows, and compact menus. Use click_area only for "
        "large, unambiguous targets. If a tap does not change the screen, do not "
        "repeat the same coordinate; choose a better point on the intended target "
        "or use navigation. For text entry, focus a field with a coordinate action "
        "first, then use direct text typing."
    )


def test_load_prompt_text_prefers_working_directory_override(tmp_path, monkeypatch):
    relative_path = "assets/prompts/state/screenshot_only_guidance.md"
    override = tmp_path / relative_path
    override.parent.mkdir(parents=True)
    override.write_text("working directory override", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    load_prompt_text.cache_clear()
    try:
        assert load_prompt_text("state/screenshot_only_guidance.md") == (
            "working directory override"
        )
    finally:
        load_prompt_text.cache_clear()
