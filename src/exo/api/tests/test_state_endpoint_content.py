"""GET /state must not expose request content."""

from exo.api.main import API
from exo.shared.types.common import CommandId, ModelId
from exo.shared.types.state import State
from exo.shared.types.tasks import TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import InstanceId

MARKER = "marker-7f3c9e"


def _api_with_running_request() -> API:
    task = TextGeneration(
        instance_id=InstanceId("inst-1"),
        command_id=CommandId("cmd-1"),
        task_params=TextGenerationTaskParams(
            model=ModelId("test-model"),
            input=[InputMessage(role="user", content=InputMessageContent(MARKER))],
        ),
    )
    api = object.__new__(API)
    api.state = State(tasks={task.task_id: task})
    return api


def test_state_leaves_out_tasks() -> None:
    api = _api_with_running_request()

    state = api.get_state()

    assert isinstance(state, State)
    assert state.tasks == {}
    assert MARKER not in state.model_dump_json()
    assert len(api.state.tasks) == 1


def test_state_path_has_no_tasks() -> None:
    api = _api_with_running_request()

    assert api.get_state("tasks") == {}
