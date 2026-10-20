"""A node that restarts doesn't try to rejoin an instance whose runner it lost."""

from collections.abc import Mapping

import anyio
import pytest

from exo.shared.types.commands import (
    DeleteInstance,
    ForwarderCommand,
    ForwarderDownloadCommand,
)
from exo.shared.types.events import Event, IndexedEvent
from exo.shared.types.state import State
from exo.shared.types.tasks import CreateRunner
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerReady,
    RunnerStatus,
)
from exo.utils.channels import Receiver, channel
from exo.worker.main import Worker
from exo.worker.tests.constants import (
    INSTANCE_1_ID,
    MODEL_A_ID,
    NODE_A,
    NODE_B,
    RUNNER_1_ID,
    RUNNER_2_ID,
)
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)


def worker_with(
    runners: Mapping[RunnerId, RunnerStatus],
) -> tuple[Worker, Receiver[ForwarderCommand]]:
    instance = get_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        node_to_runner={NODE_A: RUNNER_1_ID, NODE_B: RUNNER_2_ID},
        runner_to_shard={
            RUNNER_1_ID: get_pipeline_shard_metadata(MODEL_A_ID, 0, world_size=2),
            RUNNER_2_ID: get_pipeline_shard_metadata(MODEL_A_ID, 1, world_size=2),
        },
    )
    event_send, _event_recv = channel[Event]()
    _index_send, index_recv = channel[IndexedEvent]()
    command_send, command_recv = channel[ForwarderCommand]()
    download_send, _download_recv = channel[ForwarderDownloadCommand]()
    worker = Worker(
        NODE_A,
        event_receiver=index_recv,
        event_sender=event_send,
        command_sender=command_send,
        download_command_sender=download_send,
        api_port=52415,
    )
    worker.state = State(
        instances={INSTANCE_1_ID: instance}, runners=runners, downloads={NODE_A: []}
    )
    return worker, command_recv


async def run_plan(
    worker: Worker, monkeypatch: pytest.MonkeyPatch
) -> list[CreateRunner]:
    created: list[CreateRunner] = []

    async def create_supervisor(task: CreateRunner) -> None:
        created.append(task)

    monkeypatch.setattr(worker, "_create_supervisor", create_supervisor)
    async with anyio.create_task_group() as tg:
        tg.start_soon(worker.plan_step)
        await anyio.sleep(0.5)
        tg.cancel_scope.cancel()
    return created


async def test_runner_from_before_a_restart_gets_its_instance_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The state still has this node's runner from before it restarted; this worker never
    # started it
    worker, commands = worker_with(
        {RUNNER_1_ID: RunnerReady(), RUNNER_2_ID: RunnerReady()}
    )

    created = await run_plan(worker, monkeypatch)

    assert created == []
    sent = [forwarded.command for forwarded in commands.collect()]
    # Once, although the planner asks again every step until the deletion arrives
    assert len(sent) == 1
    assert isinstance(sent[0], DeleteInstance)
    assert sent[0].instance_id == INSTANCE_1_ID


async def test_new_instance_gets_a_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    worker, commands = worker_with({})

    created = await run_plan(worker, monkeypatch)

    assert created[0].bound_instance.bound_runner_id == RUNNER_1_ID
    assert commands.collect() == []


async def test_runner_this_worker_lost_is_recreated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker, commands = worker_with(
        {RUNNER_1_ID: RunnerFailed(diagnostics=[]), RUNNER_2_ID: RunnerReady()}
    )
    worker._started_runners.add(RUNNER_1_ID)  # pyright: ignore[reportPrivateUsage]

    created = await run_plan(worker, monkeypatch)

    assert created[0].bound_instance.bound_runner_id == RUNNER_1_ID
    assert commands.collect() == []
