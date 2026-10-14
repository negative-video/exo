# pyright: reportPrivateUsage=false
"""The thinking budget as a logits processor, without a model: masking, the
output it leaves for the parsers, and the token history the batched step gives
processors."""

import json
from collections.abc import Callable, Generator
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.generate import GenerationBatch, SequenceStateMachine
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.tool_parsers import qwen3_coder

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.runner_response import GenerationResponse, ToolCallResponse
from exo.worker.engines.mlx.generator.generate import thinking_budget_processor
from exo.worker.engines.mlx.generator.thinking_budget import THINKING_BUDGET_NOTE
from exo.worker.engines.mlx.patches.opt_batch_gen import _patched_step
from exo.worker.runner.llm_inference.model_output_parsers import (
    parse_thinking_models,
    parse_tool_calls,
)
from exo.worker.runner.llm_inference.tool_parsers import make_mlx_parser

Processor = Callable[[mx.array, mx.array], mx.array]

THINK_START = 0
THINK_END = 1
THOUGHT = 2
EOS = 3
VOCAB_SIZE = 16
TEXT = {
    THINK_START: "<think>",
    THINK_END: "</think>",
    THOUGHT: " Maybe.",
    EOS: "",
    4: "\n\n",
    5: "<tool_call>",
    6: "\n<function=search_documents>",
    7: "\n<parameter=query>",
    8: "\nthinking budget",
    9: "\n</parameter>",
    10: "\n</function>",
    11: "\n</tool_call>",
    14: THINKING_BUDGET_NOTE[:20],
    15: THINKING_BUDGET_NOTE[20:],
}
NOTE = [14, 15]
TOOL_CALL = [4, 5, 6, 7, 8, 9, 10, 11, EOS]
# MiMo's generation prompt: the model opens <think> itself.
PROMPT = "<|im_start|>user\nWhere is it?<|im_end|>\n<|im_start|>assistant\n"
PROMPT_IN_THINKING = PROMPT + "<think>\n"
# Token ids for the end of the prompt; they only need to differ from the above.
PROMPT_TAIL = [12, 13]


class _FakeTokenizer:
    think_start = "<think>"
    think_end = "</think>"

    def __init__(self, has_thinking: bool):
        self.think_start_tokens = [THINK_START] if has_thinking else None
        self.think_end_tokens = [THINK_END] if has_thinking else None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert text == THINKING_BUDGET_NOTE
        assert not add_special_tokens
        return NOTE


def _tokenizer(has_thinking: bool = True) -> TokenizerWrapper:
    return cast(Any, _FakeTokenizer(has_thinking))  # pyright: ignore[reportAny]


def _task(
    thinking_budget: int | None,
    enable_thinking: bool | None = True,
    max_output_tokens: int | None = None,
) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("test-model"),
        input=[InputMessage(role="user", content=InputMessageContent("Where is it?"))],
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        max_output_tokens=max_output_tokens,
    )


def _processor(
    budget: int, prompt: str = PROMPT, max_output_tokens: int | None = None
) -> Processor:
    task = _task(budget, max_output_tokens=max_output_tokens)
    processor = thinking_budget_processor(_tokenizer(), task, prompt, None)
    assert processor is not None
    return processor


def _logits_preferring(token: int) -> mx.array:
    logits = mx.full((1, VOCAB_SIZE), -5.0)
    logits[..., token] = 5.0
    return logits


def _searching_model(generated: list[int]) -> int:
    """Opens thinking, thinks until its thinking is closed, then calls a tool."""
    if not generated:
        return THINK_START
    if THINK_END not in generated:
        return THOUGHT
    return TOOL_CALL[len(generated) - generated.index(THINK_END) - 1]


def _decode(
    processor: Processor, model: Callable[[list[int]], int], prompt_tail: list[int]
) -> list[int]:
    """Greedy decode, passing the processor the history the generators pass:
    the end of the prompt plus every token fed in so far."""
    generated: list[int] = []
    while len(generated) < 64 and (not generated or generated[-1] != EOS):
        logits = processor(
            mx.array(prompt_tail + generated), _logits_preferring(model(generated))
        )
        generated.append(int(mx.argmax(logits, axis=-1).item()))
    return generated


def _responses(tokens: list[int]) -> Generator[GenerationResponse]:
    for index, token in enumerate(tokens):
        yield GenerationResponse(
            text=TEXT[token],
            token=token,
            finish_reason="stop" if index == len(tokens) - 1 else None,
            usage=None,
        )


@pytest.mark.parametrize(
    "prompt_tail",
    # SequentialGenerator's first call sees one prompt token, BatchGenerator's two.
    [PROMPT_TAIL[-1:], PROMPT_TAIL],
    ids=["sequential", "batched"],
)
def test_forced_close_parses_into_reasoning_content_and_tool_call(
    prompt_tail: list[int],
):
    generated = _decode(_processor(5), _searching_model, prompt_tail)
    assert generated == [THINK_START] + [THOUGHT] * 5 + NOTE + [THINK_END] + TOOL_CALL

    parsed = list(
        parse_tool_calls(
            parse_thinking_models(
                _responses(generated),
                "<think>",
                "</think>",
                starts_in_thinking=False,
            ),
            make_mlx_parser(
                qwen3_coder.tool_call_start,
                qwen3_coder.tool_call_end,
                qwen3_coder.parse_tool_call,
            ),
            tools=None,
        )
    )
    text = [r for r in parsed if isinstance(r, GenerationResponse)]
    reasoning = "".join(r.text for r in text if r.is_thinking)
    content = "".join(r.text for r in text if not r.is_thinking)
    tool_calls = [
        (call.name, json.loads(call.arguments))
        for r in parsed
        if isinstance(r, ToolCallResponse)
        for call in r.tool_calls
    ]
    assert reasoning == TEXT[THOUGHT] * 5 + THINKING_BUDGET_NOTE
    assert content == "\n\n"
    assert tool_calls == [("search_documents", {"query": "thinking budget"})]


def test_thinking_opened_by_the_prompt_is_closed_at_the_budget():
    def model(generated: list[int]) -> int:
        return THOUGHT if THINK_END not in generated else EOS

    generated = _decode(_processor(3, PROMPT_IN_THINKING), model, PROMPT_TAIL)
    assert generated == [THOUGHT] * 3 + NOTE + [THINK_END, EOS]


def test_budget_is_lowered_to_leave_room_under_the_output_cap():
    def model(generated: list[int]) -> int:
        return THOUGHT if THINK_END not in generated else EOS

    # With a 10-token cap, half of it is kept for the answer.
    processor = _processor(100, PROMPT_IN_THINKING, max_output_tokens=10)
    generated = _decode(processor, model, PROMPT_TAIL)
    assert generated == [THOUGHT] * 5 + NOTE + [THINK_END, EOS]


def test_forced_step_leaves_only_the_forced_token():
    processor = _processor(1, PROMPT_IN_THINKING)
    processor(mx.array(PROMPT_TAIL), _logits_preferring(THOUGHT))
    forced = processor(mx.array(PROMPT_TAIL + [THOUGHT]), _logits_preferring(EOS))

    assert int(mx.sum(forced > -1e8).item()) == 1
    logprobs = forced - mx.logsumexp(forced, axis=-1, keepdims=True)
    sampler = make_sampler(temp=1.0)
    samples = [int(sampler(logprobs).item()) for _ in range(50)]
    assert samples == [NOTE[0]] * 50


def test_natural_close_before_the_budget_leaves_logits_untouched():
    processor = _processor(100)
    script = [THINK_START, THOUGHT, THOUGHT, THINK_END, 4, EOS]
    history = PROMPT_TAIL[:]
    for token in script:
        logits = _logits_preferring(token)
        assert processor(mx.array(history), logits) is logits
        history.append(token)


@pytest.mark.parametrize(
    "thinking_budget,enable_thinking", [(None, True), (0, True), (100, False)]
)
def test_no_processor_without_a_budget(
    thinking_budget: int | None, enable_thinking: bool
):
    task = _task(thinking_budget, enable_thinking)
    assert thinking_budget_processor(_tokenizer(), task, PROMPT, None) is None


def test_no_processor_for_a_model_without_thinking():
    task = _task(100)
    assert thinking_budget_processor(_tokenizer(False), task, PROMPT, None) is None


class _NextTokenModel(nn.Module):
    """Prefers the token after the one fed in."""

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        return mx.eye(VOCAB_SIZE)[(x + 1) % VOCAB_SIZE]


def _greedy(logprobs: mx.array) -> mx.array:
    return mx.argmax(logprobs, axis=-1)


def _recorder(histories: list[list[int]]) -> Processor:
    def record(history: mx.array, logits: mx.array) -> mx.array:
        histories.append(cast(list[int], history.tolist()))
        return logits

    return record


def test_batched_step_gives_processors_the_token_fed_in(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(GenerationBatch, "_step", _patched_step)
    histories: list[list[int]] = []
    batch = GenerationBatch(
        model=_NextTokenModel(),
        uids=[0],
        inputs=mx.array([5]),
        prompt_cache=[],
        tokens=[[4]],
        samplers=[_greedy],
        fallback_sampler=_greedy,
        logits_processors=[[_recorder(histories)]],
        state_machines=[SequenceStateMachine()],
        max_tokens=[16],
    )
    batch._step()
    batch._step()
    assert histories == [[4, 5], [4, 5, 6], [4, 5, 6, 7]]


def test_batched_step_handles_a_sequence_without_processors(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(GenerationBatch, "_step", _patched_step)
    histories: list[list[int]] = []
    # mlx-lm stores None for a sequence without processors when it shares a
    # prompt batch with one that has them.
    processors: list[list[Processor] | None] = [None, [_recorder(histories)]]
    batch = GenerationBatch(
        model=_NextTokenModel(),
        uids=[0, 1],
        inputs=mx.array([5, 9]),
        prompt_cache=[],
        tokens=[[4], [8]],
        samplers=[_greedy, _greedy],
        fallback_sampler=_greedy,
        logits_processors=cast(Any, processors),  # pyright: ignore[reportAny]
        state_machines=[SequenceStateMachine(), SequenceStateMachine()],
        max_tokens=[16, 16],
    )
    batch._step()
    assert histories == [[8, 9], [8, 9, 10]]
