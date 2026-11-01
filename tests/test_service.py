import asyncio
import json
import sys
from pathlib import Path

import pytest

from src.tts.config import Settings
from src.tts.exceptions import SpeechError
from src.tts.schemas import SpeechItem
from src.tts.service import SpeechService

ITEM = SpeechItem(text="Hello", language="English", speaker_name="Ryan")


def make_service(tmp_path, mode="ok", timeout=5):
    script = tmp_path / "worker.py"
    marker = tmp_path / "marker.json"
    script.write_text("""import json, os, sys, time
from pathlib import Path
source, target = map(Path, sys.argv[-2:])
Path(sys.argv[1]).write_text(json.dumps(dict(pid=os.getpid(), directory=str(source.parent))))
if sys.argv[2] == 'sleep': time.sleep(30)
if sys.argv[2] == 'crash': sys.exit(1)
target.write_text(json.dumps(dict(model='test', results=[])))
""")
    return SpeechService(
        Settings(timeout=timeout), [sys.executable, str(script), str(marker), mode]
    ), marker


async def wait_marker(marker):
    for _ in range(300):
        if marker.exists():
            return json.loads(marker.read_text())
        await asyncio.sleep(0.01)
    raise AssertionError("Worker did not start")


def assert_clean(marker, service):
    state = json.loads(marker.read_text())
    assert not Path(state["directory"]).exists()
    import os

    with pytest.raises(ProcessLookupError):
        os.kill(state["pid"], 0)
    assert not service.busy


async def test_success(tmp_path):
    service, marker = make_service(tmp_path)
    assert (await service([ITEM]))["model"] == "test"
    assert_clean(marker, service)
    assert (await service([ITEM]))["model"] == "test"


async def test_timeout_and_busy(tmp_path):
    service, marker = make_service(tmp_path, "sleep", 0.2)
    task = asyncio.create_task(service([ITEM]))
    await wait_marker(marker)
    with pytest.raises(SpeechError) as busy:
        await service([ITEM])
    assert busy.value.status == 503
    with pytest.raises(SpeechError) as timeout:
        await task
    assert timeout.value.status == 504
    assert_clean(marker, service)
    service.command[-1] = "ok"
    assert (await service([ITEM]))["model"] == "test"


@pytest.mark.parametrize("shutdown", [False, True])
async def test_cancel_and_shutdown(tmp_path, shutdown):
    service, marker = make_service(tmp_path, "sleep")
    task = asyncio.create_task(service([ITEM]))
    await wait_marker(marker)
    if shutdown:
        await service.close()
    else:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert_clean(marker, service)


async def test_crash(tmp_path):
    service, marker = make_service(tmp_path, "crash")
    with pytest.raises(SpeechError) as error:
        await service([ITEM])
    assert error.value.status == 500
    assert_clean(marker, service)


@pytest.mark.parametrize(
    "items,settings",
    [
        ([], {}),
        ([ITEM] * 9, {}),
        ([ITEM], {"max_text_chars": 2}),
        ([ITEM] * 2, {"max_total_chars": 5}),
    ],
)
async def test_limits(items, settings):
    service = SpeechService(Settings(**settings))
    with pytest.raises(SpeechError) as error:
        await service(items)
    assert error.value.status == 422
    assert not service.busy


async def test_disconnect(tmp_path):
    from starlette.requests import Request

    from src.tts.router import speech

    service, marker = make_service(tmp_path, "sleep")

    async def receive():
        return {"type": "http.disconnect"}

    request = Request({"type": "http", "method": "POST", "path": "/v1/speech"}, receive)
    with pytest.raises(asyncio.CancelledError):
        await speech([ITEM], request, service)
    assert_clean(marker, service)


async def test_output_limit(tmp_path):
    service, marker = make_service(tmp_path)
    service.settings.max_result_bytes = 1
    with pytest.raises(SpeechError) as error:
        await service([ITEM])
    assert error.value.status == 413
    assert_clean(marker, service)
