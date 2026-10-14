from exo.shared.types.text_generation import (
    THINKING_BUDGET_BY_EFFORT,
    requested_thinking_budget,
)


def test_nothing_requested_leaves_the_node_default() -> None:
    assert requested_thinking_budget(None, None) is None


def test_reasoning_effort_sets_its_level() -> None:
    assert requested_thinking_budget(None, "minimal") == 2048
    assert requested_thinking_budget(None, "low") == 4096
    assert requested_thinking_budget(None, "medium") == 12288
    assert requested_thinking_budget(None, "high") == 20480
    assert requested_thinking_budget(None, "xhigh") == 24576


def test_thinking_off_has_no_budget() -> None:
    assert requested_thinking_budget(None, "none") is None


def test_explicit_budget_wins_over_reasoning_effort() -> None:
    assert requested_thinking_budget(300, "high") == 300
    assert requested_thinking_budget(0, "low") == 0


def test_every_level_stays_under_the_output_cap_with_room_for_an_answer() -> None:
    assert max(THINKING_BUDGET_BY_EFFORT.values()) <= 24576
