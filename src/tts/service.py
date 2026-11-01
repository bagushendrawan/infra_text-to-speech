import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.concurrency import run_in_threadpool

from src.tts.exceptions import SpeechError


class SpeechService:
    def __init__(self, settings, command=None):
        self.settings = settings
        self.command = command or [sys.executable, "-m", "src.tts.worker"]
        self.busy = False
        self.task = None
        self.closed = False

    def validate(self, items):
        s = self.settings
        if not 1 <= len(items) <= s.max_items:
            raise SpeechError(422, f"Batch must contain 1 to {s.max_items} items")
        for i, item in enumerate(items):
            if len(item.text) > s.max_text_chars or len(item.instruct) > s.max_instruction_chars:
                raise SpeechError(422, f"Item {i} exceeds text/instruction limit")
        if sum(len(x.text) for x in items) > s.max_total_chars:
            raise SpeechError(422, "Batch exceeds total text limit")

    async def close(self):
        self.closed = True
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def __call__(self, items):
        self.validate(items)
        if self.busy or self.closed:
            raise SpeechError(503, "Speech worker unavailable")
        self.busy = True
        self.task = asyncio.current_task()
        process = None
        temporary = None
        try:
            temporary = TemporaryDirectory(prefix="tts-")
            work = Path(temporary.name)
            source, target = work / "input.json", work / "result.json"
            payload = dict(
                settings=self.settings.model_dump(), items=[x.model_dump() for x in items]
            )
            await run_in_threadpool(source.write_text, json.dumps(payload), encoding="utf-8")
            # Shield process creation so cancellation cannot orphan a newly spawned worker.
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *self.command,
                    str(source),
                    str(target),
                    start_new_session=True,
                    cwd=Path(__file__).resolve().parents[2],
                )
            )
            try:
                process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                process = await spawn
                raise
            try:
                await asyncio.wait_for(process.wait(), self.settings.timeout)
            except TimeoutError as exc:
                raise SpeechError(504, "Speech generation timed out") from exc
            if process.returncode:
                raise SpeechError(500, "Speech worker failed; see server logs")

            def read_result():
                if target.stat().st_size > self.settings.max_result_bytes:
                    raise SpeechError(413, "Generated result exceeds size limit")
                return json.loads(target.read_text(encoding="utf-8"))

            result = await run_in_threadpool(read_result)
            if "error" in result:
                raise SpeechError(result["status"], result["error"])
            return result
        except (OSError, ValueError) as exc:
            raise SpeechError(500, "Speech worker failed; see server logs") from exc
        finally:
            if process is not None and process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), 5)
                except TimeoutError:
                    os.killpg(process.pid, signal.SIGKILL)
                    await process.wait()
            try:
                if temporary is not None:
                    await run_in_threadpool(temporary.cleanup)
            finally:
                self.task = None
                self.busy = False
