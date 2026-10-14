"""State machine behind the thinking budget, driven without a model."""

from collections.abc import Callable

from exo.worker.engines.mlx.generator.thinking_budget import (
    ANSWER_RESERVE_TOKENS,
    ThinkingBudget,
    budget_within_output_cap,
)

THINK_START = 1
THINK_END = 2
WORD = 3
ANSWER = 4
END_A = 5
END_B = 6
NOTE = [7, 8, 9]


def _budget(
    budget: int,
    starts_in_thinking: bool,
    think_end: list[int] | None = None,
) -> ThinkingBudget:
    return ThinkingBudget(
        budget=budget,
        think_start_tokens=[THINK_START],
        think_end_tokens=think_end or [THINK_END],
        note_tokens=NOTE,
        starts_in_thinking=starts_in_thinking,
    )


def _generate(
    budget: ThinkingBudget, model: Callable[[list[int]], int], steps: int
) -> list[int]:
    """Sample `model`'s choice each step unless the budget forces a token."""
    generated: list[int] = []
    for _ in range(steps):
        forced = budget.forced_token()
        token = forced if forced is not None else model(generated)
        budget.observe(token)
        generated.append(token)
    return generated


def _thinks_until_closed(
    think_end: list[int], opens_thinking: bool
) -> Callable[[list[int]], int]:
    """A model that thinks forever and answers once its thinking is closed."""

    def model(generated: list[int]) -> int:
        if opens_thinking and not generated:
            return THINK_START
        for start in range(len(generated) - len(think_end) + 1):
            if generated[start : start + len(think_end)] == think_end:
                return ANSWER
        return WORD

    return model


def test_thinking_opened_by_the_prompt_is_closed_at_the_budget():
    budget = _budget(5, starts_in_thinking=True)
    model = _thinks_until_closed([THINK_END], opens_thinking=False)
    generated = _generate(budget, model, steps=11)
    assert generated == [WORD] * 5 + NOTE + [THINK_END, ANSWER, ANSWER]
    assert budget.phase == "finished"


def test_thinking_opened_in_generation_is_closed_at_the_budget():
    budget = _budget(5, starts_in_thinking=False)
    model = _thinks_until_closed([THINK_END], opens_thinking=True)
    generated = _generate(budget, model, steps=12)
    assert generated == [THINK_START] + [WORD] * 5 + NOTE + [THINK_END, ANSWER, ANSWER]


def test_natural_close_before_the_budget_is_left_alone():
    budget = _budget(5, starts_in_thinking=False)
    script = [THINK_START, WORD, WORD, WORD, WORD, THINK_END] + [ANSWER] * 6

    def model(generated: list[int]) -> int:
        return script[len(generated)]

    assert _generate(budget, model, steps=len(script)) == script
    assert budget.phase == "finished"


def test_budget_forces_the_note_and_every_token_of_a_multi_token_think_end():
    budget = _budget(4, starts_in_thinking=True, think_end=[END_A, END_B])
    model = _thinks_until_closed([END_A, END_B], opens_thinking=False)
    generated = _generate(budget, model, steps=10)
    assert generated == [WORD] * 4 + NOTE + [END_A, END_B, ANSWER]
    assert budget.phase == "finished"


def test_budget_completes_a_think_end_the_model_already_started():
    budget = _budget(4, starts_in_thinking=True, think_end=[END_A, END_B])
    # The model's fourth thinking token happens to be the first think-end token,
    # so the close finishes that sequence instead of writing the note.
    script = [WORD, WORD, WORD, END_A]

    def model(generated: list[int]) -> int:
        return script[len(generated)] if len(generated) < len(script) else ANSWER

    generated = _generate(budget, model, steps=6)
    assert generated == [WORD, WORD, WORD, END_A, END_B, ANSWER]


def test_no_interference_after_thinking_closes():
    budget = _budget(2, starts_in_thinking=True)
    # A second thinking block after the forced close is not limited.
    after_close = [THINK_START] + [WORD] * 10

    def model(generated: list[int]) -> int:
        if THINK_END not in generated:
            return WORD
        return after_close[len(generated) - generated.index(THINK_END) - 1]

    generated = _generate(budget, model, steps=2 + len(NOTE) + 1 + len(after_close))
    assert generated == [WORD, WORD] + NOTE + [THINK_END] + after_close


def test_thinking_tokens_counted_at_the_trigger():
    budget = _budget(3, starts_in_thinking=False)
    for token in [THINK_START, WORD, WORD]:
        budget.observe(token)
        assert budget.forced_token() is None
    budget.observe(WORD)
    assert budget.thinking_tokens == 3
    assert budget.forced_token() == NOTE[0]


def test_close_without_an_open_ends_tracking():
    budget = _budget(1, starts_in_thinking=False)
    for token in [WORD, THINK_END, THINK_START, WORD, WORD]:
        budget.observe(token)
        assert budget.forced_token() is None
    assert budget.phase == "finished"


def test_budget_leaves_room_for_the_answer_under_the_output_cap():
    # The default cap leaves every reasoning_effort level as it is.
    assert budget_within_output_cap(12288, 32168) == 12288
    assert budget_within_output_cap(24576, 32168) == 24576
    # A budget that would leave too little room is lowered.
    assert budget_within_output_cap(32768, 32168) == 32168 - ANSWER_RESERVE_TOKENS
    assert budget_within_output_cap(12288, 10000) == 10000 - ANSWER_RESERVE_TOKENS
    # A small cap keeps half for the answer.
    assert budget_within_output_cap(12288, 6000) == 3000
    assert budget_within_output_cap(12288, 1) == 1
