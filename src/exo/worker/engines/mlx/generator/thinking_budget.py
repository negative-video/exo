"""Per-request ceiling on the number of tokens a model may spend thinking.

`ThinkingBudget` follows one request's generated tokens. Once the model has
thought for `budget` tokens without closing its thinking block, it names the
tokens to force, one per step: a short note, then the think-end sequence.
Generation then continues freely to the answer or a tool call. Every decision
comes from the token history alone, so every pipeline rank that samples the
same tokens forces the same ones.
"""

from collections.abc import Sequence
from typing import Literal, final

ThinkingPhase = Literal["before_thinking", "thinking", "closing", "finished"]
CloseReason = Literal["budget", "answer_intent"]

# Forced before the think-end tokens to steer the model toward acting on what
# it has instead of re-checking it. Answering comes first: text written after
# the forced close is visible reply, which the budget does not cut. Earlier
# thinking is not sent back on later turns, so otherwise the note asks for a
# one-line status in the visible reply before the next tool call.
THINKING_BUDGET_NOTE = (
    "\n\n(Thinking budget reached. If you were about to write the answer, "
    "write it now in your reply, in full, and make no more tool calls. "
    "Otherwise write one line, 'Established: ...; still open: ...', then run "
    "the pending calculation in Python.)\n"
)

# A round whose thinking opens by announcing the answer ("Now let me write the
# answer") is closed at once, so the model writes the answer as visible reply
# instead of drafting it in thinking, where the budget cuts it and the note's
# "otherwise" branch reopens the work. Only the first ANSWER_INTENT_WINDOW
# thinking tokens count; the phrases are matched as token sequences.
ANSWER_INTENT_PHRASES = (
    " write the answer",
    " write the final answer",
    " assemble the answer",
    "Write the answer",
    " I have everything",
    "I have everything",
)
ANSWER_INTENT_WINDOW = 256
ANSWER_INTENT_NOTE = (
    "\n\n(You have what you need. Write the answer now in your reply, in full, "
    "and make no more tool calls.)\n"
)

# Output tokens a forced close leaves free for the note, the answer or a tool
# call. MiMo's longest answer after thinking has been about 3.5k tokens.
ANSWER_RESERVE_TOKENS = 4096


def budget_within_output_cap(budget: int, max_tokens: int) -> int:
    """Lower `budget` so thinking closes with room left for the answer, so a
    response is never cut off mid-thought by the output cap."""
    room = max(max_tokens - ANSWER_RESERVE_TOKENS, max_tokens // 2)
    return max(1, min(budget, room))


@final
class ThinkingBudget:
    def __init__(
        self,
        budget: int,
        think_start_tokens: Sequence[int],
        think_end_tokens: Sequence[int],
        note_tokens: Sequence[int],
        starts_in_thinking: bool,
        intent_sequences: Sequence[Sequence[int]] = (),
        intent_note_tokens: Sequence[int] = (),
    ):
        assert budget > 0
        assert len(think_start_tokens) > 0
        assert len(think_end_tokens) > 0
        self.budget = budget
        self._think_start_tokens = tuple(think_start_tokens)
        self._think_end_tokens = tuple(think_end_tokens)
        self._note_tokens = tuple(note_tokens)
        self._intent_sequences = tuple(tuple(q) for q in intent_sequences if q)
        self._intent_note_tokens = tuple(intent_note_tokens)
        self.close_reason: CloseReason | None = None
        self.phase: ThinkingPhase = (
            "thinking" if starts_in_thinking else "before_thinking"
        )
        self.thinking_tokens = 0
        self._closing_tokens: tuple[int, ...] = ()
        self._closing_position = 0
        self._recent_tokens: list[int] = []
        self._recent_window = max(
            len(self._think_start_tokens),
            len(self._think_end_tokens),
            *(len(q) for q in self._intent_sequences),
        )

    def observe(self, token: int) -> None:
        """Advance the state by one generated token."""
        if self.phase == "finished":
            return
        self._recent_tokens.append(token)
        del self._recent_tokens[: -self._recent_window]

        if self.phase == "before_thinking":
            if self._recent_tokens_end_with(self._think_end_tokens):
                # A block we never saw open was closed; nothing left to limit.
                self.phase = "finished"
            elif self._recent_tokens_end_with(self._think_start_tokens):
                self.phase = "thinking"
                self._recent_tokens.clear()
            return

        if self._recent_tokens_end_with(self._think_end_tokens):
            self.phase = "finished"
            return

        if self.phase == "closing":
            if token == self._closing_tokens[self._closing_position]:
                self._closing_position += 1
            return

        self.thinking_tokens += 1
        if self.thinking_tokens <= ANSWER_INTENT_WINDOW and any(
            self._recent_tokens_end_with(q) for q in self._intent_sequences
        ):
            self._close("answer_intent", self._intent_note_tokens)
        elif self.thinking_tokens >= self.budget:
            self._close("budget", self._note_tokens)

    def _close(self, reason: CloseReason, note: tuple[int, ...]) -> None:
        self.phase = "closing"
        self.close_reason = reason
        prefix_length = self._think_end_prefix_length()
        # Finish a think-end sequence the model already started; otherwise
        # write the note first.
        self._closing_tokens = (
            self._think_end_tokens[prefix_length:]
            if prefix_length > 0
            else note + self._think_end_tokens
        )

    def forced_token(self) -> int | None:
        """The token the next step must produce, or None to sample freely."""
        if self.phase != "closing":
            return None
        return self._closing_tokens[self._closing_position]

    def _recent_tokens_end_with(self, sequence: tuple[int, ...]) -> bool:
        return tuple(self._recent_tokens[-len(sequence) :]) == sequence

    def _think_end_prefix_length(self) -> int:
        for length in range(len(self._think_end_tokens) - 1, 0, -1):
            if self._recent_tokens_end_with(self._think_end_tokens[:length]):
                return length
        return 0
