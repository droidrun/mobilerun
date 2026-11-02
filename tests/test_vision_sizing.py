"""Tests for the per-model vision resize policy (issue #365)."""

from types import SimpleNamespace

from mobilerun.agent.utils.vision_sizing import (
    VisionResizePolicy,
    model_effective_dims,
)
from mobilerun.tools.helpers.images import anthropic_resized_size


def test_anthropic_resized_size_standard_budget():
    # Tall phone screenshot, standard 1568 budget → long edge clamped to 1568.
    assert anthropic_resized_size(922, 2048, 1568, 1568) == (706, 1568)
    # 16:9 case where the token budget binds tighter than the 1568 edge.
    assert anthropic_resized_size(1080, 1920, 1568, 1568) == (819, 1456)


def test_anthropic_resized_size_highres_budget_keeps_2048():
    # High-res budget (Opus 4.7+): a 922x2048 image fits unchanged.
    assert anthropic_resized_size(922, 2048, 2576, 4784) == (922, 2048)


def test_model_effective_dims_per_provider():
    # Standard Anthropic model downsizes; high-res Anthropic + others do not.
    assert model_effective_dims("claude-sonnet-4-6", 1080, 2400) == (706, 1568)
    assert model_effective_dims("claude-opus-4-7", 1080, 2400) == (922, 2048)
    assert model_effective_dims("gpt-5.5", 1080, 2400) == (922, 2048)
    assert model_effective_dims("gemini-3.5-flash-low", 1080, 2400) == (922, 2048)
    # Unknown Anthropic id is treated as STANDARD (never assume high-res).
    assert model_effective_dims("claude-future-9", 1080, 2400) == (706, 1568)


def test_claude_5_models_use_high_resolution_budget():
    for model in (
        "claude-fable-5-1",
        "claude-opus-5",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-fable-5",
    ):
        assert model_effective_dims(model, 1080, 2400) == (922, 2048)


def test_sonnet_5_grounds_on_a_1568_long_edge():
    assert model_effective_dims("claude-sonnet-5", 1080, 2400) == (706, 1568)
    assert model_effective_dims("claude-sonnet-5", 2560, 1600) == (1568, 980)
    assert model_effective_dims("claude-sonnet-5", 720, 1280) == (720, 1280)
    assert VisionResizePolicy(["claude-opus-5-5", "claude-sonnet-5"]).effective_dims(
        1080, 2400
    ) == (706, 1568)


def test_unregistered_mythos_model_retains_high_resolution_budget():
    assert model_effective_dims("claude-mythos-5", 1080, 2400) == (922, 2048)


def test_policy_single_model():
    assert VisionResizePolicy(["claude-sonnet-4-6"]).effective_dims(1080, 2400) == (
        706,
        1568,
    )
    assert VisionResizePolicy(["gpt-5.5"]).effective_dims(1080, 2400) == (922, 2048)


def test_policy_composite_takes_smallest():
    # Mixed manager/executor models → declare the smallest so both are correct.
    policy = VisionResizePolicy(["claude-opus-4-7", "claude-sonnet-4-6"])
    assert policy.effective_dims(1080, 2400) == (706, 1568)
    # All high-res / non-downsizing → stays at 2048.
    assert VisionResizePolicy(["claude-opus-4-7", "gpt-5.5"]).effective_dims(
        1080, 2400
    ) == (922, 2048)


def test_policy_empty_defaults_to_2048_fit():
    assert VisionResizePolicy([]).effective_dims(1080, 2400) == (922, 2048)


def test_policy_max_side_cap_shrinks_further():
    policy = VisionResizePolicy(["claude-sonnet-4-6"], max_side_cap=1280)
    w, h = policy.effective_dims(1080, 2400)
    assert max(w, h) == 1280


def test_policy_from_llms_reads_model_attr():
    llms = [
        SimpleNamespace(model="claude-sonnet-4-6"),
        SimpleNamespace(model="gpt-5.5"),
    ]
    policy = VisionResizePolicy.from_llms(llms)
    # min across sonnet(1568) and gpt(2048) → 1568.
    assert policy.effective_dims(1080, 2400) == (706, 1568)
