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

# Forced before the think-end tokens to steer the model toward acting on what
# it has instead of re-checking it. Earlier thinking is not sent back on later
# turns, so the note also asks for a one-line status in the visible reply.
THINKING_BUDGET_NOTE = (
    "\n\n(Thinking budget reached. In your reply, first write one line: "
    "'Established: ...; still open: ...'. Then run the pending calculation "
    "in Python, or write the answer.)\n"
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
    ):
        assert budget > 0
        assert len(think_start_tokens) > 0
        assert len(think_end_tokens) > 0
        self.budget = budget
        self._think_start_tokens = tuple(think_start_tokens)
        self._think_end_tokens = tuple(think_end_tokens)
        self._note_tokens = tuple(note_tokens)
        self.phase: ThinkingPhase = (
            "thinking" if starts_in_thinking else "before_thinking"
        )
        self.thinking_tokens = 0
        self._closing_tokens: tuple[int, ...] = ()
        self._closing_position = 0
        self._recent_tokens: list[int] = []
        self._recent_window = max(
            len(self._think_start_tokens), len(self._think_end_tokens)
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
        if self.thinking_tokens >= self.budget:
            self.phase = "closing"
            prefix_length = self._think_end_prefix_length()
            # Finish a think-end sequence the model already started; otherwise
            # write the note first.
            self._closing_tokens = (
                self._think_end_tokens[prefix_length:]
                if prefix_length > 0
                else self._note_tokens + self._think_end_tokens
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
