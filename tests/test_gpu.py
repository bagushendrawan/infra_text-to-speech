"""Opt-in real CUDA tests: TTS_RUN_GPU=1 uv run --extra inference pytest tests/test_gpu.py."""

import base64
import io
import os
import wave

import pytest

from src.tts.config import Settings
from src.tts.schemas import SpeechItem
from src.tts.service import SpeechService

pytestmark = pytest.mark.skipif(os.getenv("TTS_RUN_GPU") != "1", reason="Opt-in CUDA test")


@pytest.mark.parametrize(
    "model,instruct",
    [
        ("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", ""),
        ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Speak cheerfully."),
    ],
)
async def test_cuda(model, instruct):
    service = SpeechService(Settings(model=model))
    try:
        result = await service(
            [
                SpeechItem(
                    text="Hello, how are you?",
                    language="English",
                    speaker_name="Ryan",
                    instruct=instruct,
                )
            ]
        )
        with wave.open(io.BytesIO(base64.b64decode(result["results"][0]["audio_base64"]))) as wav:
            assert wav.getnframes() > 0
    finally:
        await service.close()
