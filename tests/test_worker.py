import base64
import io
import wave

import numpy as np
import pytest

from src.tts.exceptions import SpeechError
from src.tts.worker import encode_audio, validate_capabilities


def test_wav_round_trip():
    result = encode_audio([np.zeros(2400, dtype=np.float32)], 24000)[0]
    assert result["duration_seconds"] == 0.1
    with wave.open(io.BytesIO(base64.b64decode(result["audio_base64"]))) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 24000)
        assert wav.getnframes() == 2400


@pytest.mark.parametrize(
    "field,value", [("language", "Indonesian"), ("speaker_name", "Unknown"), ("instruct", "Happy")]
)
def test_capability_rejection(field, value):
    items = [
        dict(text="Hi", language="English", speaker_name="Ryan", instruct=""),
        dict(text="Bye", language="English", speaker_name="Ryan", instruct=""),
    ]
    items[1][field] = value
    with pytest.raises(SpeechError, match="Item 1") as error:
        validate_capabilities(items, ["english"], ["ryan"], False)
    assert error.value.status == 422


def test_normalization_and_instruction_support():
    items = [dict(text="Hi", language="ENGLISH", speaker_name="Ryan", instruct="Happy")]
    validate_capabilities(items, ["english"], ["ryan"], True)
    assert items[0]["language"] == "english"
    assert items[0]["speaker_name"] == "ryan"
